"""Resumable evaluation of independent paragraph selectors and two Qwen readers."""

import argparse
import hashlib
import json
import os
import platform
import sqlite3
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from qwen_hotpot_data import (
    ABSTENTION, CONDITIONS, DENSE, MODELS, SELECTORS, TEMPLATES, answer_scores,
    checkpoint_info, clean_answer, file_hash, load_json, make_messages,
    prepare, select_indices, selector_text, write_csv, write_json,
)

DEFAULT_CHECKPOINT = Path(__file__).parent / "models" / "minilm-hotpotqa-best"
WARMUPS = 1
PILOT_START_CAP = 64


def open_store(out):
    db = sqlite3.connect(out / "evaluation.sqlite3", timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rankings (
            selector TEXT, qid TEXT, payload TEXT NOT NULL, PRIMARY KEY (selector, qid));
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY, vector BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS answers (
            stage TEXT, model TEXT, cap INTEGER, template TEXT, condition TEXT, qid TEXT, payload TEXT NOT NULL,
            PRIMARY KEY (stage, model, cap, template, condition, qid));
    """)
    return db


def pin_config(db, key, value):
    # Check and record provenance atomically so resumed data cannot mix configurations.
    encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    with db:
        db.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?)", (key, encoded))
        stored, = db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if stored != encoded:
            raise ValueError(f"Configuration changed for {key}. Use the original configuration or a new output directory.")


def software():
    result = {"python": platform.python_version()}
    for name in ("torch", "transformers", "sentence-transformers", "huggingface-hub", "tokenizers",
                 "safetensors", "numpy", "datasets", "scipy", "matplotlib"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def code_hash():
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in ("evaluate_qwen_hotpot.py", "qwen_hotpot_data.py")}


def hardware(torch, device):
    cpu = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        with cpuinfo.open(encoding="utf-8") as source:
            cpu = next((line.partition(":")[2].strip() for line in source if line.startswith("model name")), cpu)
    details = {"device": device, "os": platform.platform(), "cpu": cpu,
               "torch_cpu_threads": torch.get_num_threads(), "cuda_runtime": torch.version.cuda}
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        details.update(gpu=props.name, vram_bytes=props.total_memory, capability=list(torch.cuda.get_device_capability(0)))
    return details


def synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_ranking(args, db, rows, selector, cross):
    from math import isfinite

    pending = [ex for ex in rows if db.execute(
        "SELECT 1 FROM rankings WHERE selector=? AND qid=?", (selector, ex["qid"])
    ).fetchone() is None]
    if not pending:
        return
    import numpy as np
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for selector ranking, or move to the GPU host.")

    if selector == "dense":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(DENSE[0], revision=DENSE[1], device=args.device).eval()
        tokenizer = model.tokenizer
        limit = model.max_seq_length
        provenance = {"model": DENSE[0], "revision": DENSE[1], "precision": str(model.dtype), "quantization": None}

        def encode(texts):
            return model.encode(texts, batch_size=10, convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=False, precision="float32")

        def score(question, passages, use_cache):
            question_vector = encode([question])[0]
            if not use_cache:
                return encode(passages) @ question_vector
            # Exact text (including title) is the input and therefore the cache key.
            keys = [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in passages]
            vectors, missing = {}, {}
            for key, passage in zip(keys, passages):
                found = db.execute("SELECT vector FROM embeddings WHERE id=?", (key,)).fetchone()
                if found is None:
                    missing[key] = passage
                else:
                    vectors[key] = np.frombuffer(found[0], dtype="<f4")
            if missing:
                for key, vector in zip(missing, encode(list(missing.values()))):
                    vectors[key] = np.asarray(vector, dtype="<f4")
                    db.execute("INSERT OR IGNORE INTO embeddings VALUES (?, ?)", (key, vectors[key].tobytes()))
            return np.stack([vectors[key] for key in keys]) @ question_vector

        def input_lengths(question, passages):
            return tokenizer([question, *passages], truncation=False, verbose=False, return_length=True)["length"]

    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint, local_files_only=True, dtype=torch.float32).to(args.device).eval()
        if model.config.num_labels != 1:
            raise ValueError("The original fine-tuned MiniLM must have one relevance logit.")
        limit = int(cross["metadata"]["MAX_LENGTH"])
        provenance = {"checkpoint": cross, "precision": "torch.float32", "quantization": None}

        def score(question, passages, use_cache):
            enc = tokenizer([question] * len(passages), passages, truncation="only_second",
                            max_length=limit, padding=True, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                return model(**enc).logits[:, 0].float().cpu().numpy()

        def input_lengths(question, passages):
            return tokenizer([question] * len(passages), passages, truncation=False,
                             verbose=False, return_length=True)["length"]

    config = {
        **provenance, "hardware": hardware(torch, args.device), "software": software(),
        "max_input_tokens": limit, "warmup_questions_per_invocation": WARMUPS,
        "timing": "Batch one question. Wall time with device synchronization, including formatting, tokenization, "
                  "forward pass, transfer and sorting all ten scores. Excludes model loading and diagnostics.",
    }
    pin_config(db, f"selector/{selector}", config)
    for index, ex in enumerate(pending, 1):
        # Ranking gets only the question and ten title/body pairs. References stay outside this interface.
        question = ex["input"]["question"]
        pairs = ex["input"]["paragraphs"]
        if index == 1:
            for _ in range(WARMUPS):
                score(question, [selector_text(*p) for p in pairs], False)
        synchronize(torch, args.device)
        start = perf_counter()
        passages = [selector_text(*p) for p in pairs]
        uncached_scores = score(question, passages, False)
        ranking = sorted(range(10), key=lambda i: (-float(uncached_scores[i]), i))
        synchronize(torch, args.device)
        uncached_seconds = perf_counter() - start
        # Cache dense vectors across questions. Use the cache for the experiment rankings, alongside uncached timing.
        if selector == "dense":
            start = perf_counter()
            with db:
                scores = score(question, passages, True)
            ranking = sorted(range(10), key=lambda i: (-float(scores[i]), i))
            synchronize(torch, args.device)
            cached_seconds = perf_counter() - start
            uncached_dense_ranking = sorted(range(10), key=lambda i: (-float(uncached_scores[i]), i))
        else:
            scores, cached_seconds, uncached_dense_ranking = uncached_scores, None, None
        if len(scores) != 10 or not all(isfinite(s) for s in scores):
            raise ValueError(f"{selector} did not score exactly ten valid candidates for {ex['qid']}")
        lengths = input_lengths(question, passages)
        question_length = lengths[0] if selector == "dense" else None
        paragraph_lengths = lengths[1:] if selector == "dense" else lengths
        payload = {"qid": ex["qid"], "selector": selector,
                   "candidate_indices": list(range(10)), "scores": [float(x) for x in scores],
                   "ranking": ranking, "question_tokens_if_separately_encoded": question_length,
                   "question_truncated_if_separately_encoded": question_length is not None and question_length > limit,
                   "untruncated_input_tokens": paragraph_lengths, "truncated_paragraph_indices": [
                       i for i, length in enumerate(paragraph_lengths) if length > limit],
                   "uncached_seconds": uncached_seconds, "cached_embedding_path_seconds": cached_seconds,
                   "cached_ranking_differs": None if uncached_dense_ranking is None else ranking != uncached_dense_ranking}
        with db:
            db.execute("INSERT OR IGNORE INTO rankings VALUES (?, ?, ?)",
                       (selector, ex["qid"], json.dumps(payload, allow_nan=False)))
        if index % 25 == 0 or index == len(pending):
            print(f"{selector}: {index}/{len(pending)} rankings", flush=True)


def get_rankings(db, examples):
    all_rows = {(qid, selector): json.loads(payload) for selector, qid, payload in db.execute("SELECT * FROM rankings")}
    if any((ex["qid"], selector) not in all_rows for ex in examples for selector in SELECTORS):
        raise ValueError("Both selectors must finish ranking the relevant split before generation.")
    return all_rows


def answer_jobs(stage, examples, ranks, frozen_template=None):
    selector_conditions = tuple(f"{selector}@2" for selector in SELECTORS)
    if stage == "test":
        combinations = [(frozen_template, condition) for condition in CONDITIONS]
    else:
        combinations = [(template, condition) for template in TEMPLATES for condition in selector_conditions]
        if stage == "pilot":
            combinations += [("direct", "full"), ("direct", "oracle")]
    for template, condition in combinations:
        for ex in examples:
            qid = ex["qid"]
            rankings = {s: ranks[qid, s]["ranking"] for s in SELECTORS}
            gold_indices = ex["reference"]["gold_indices"] if condition == "oracle" else None
            indices = select_indices(condition, rankings, len(ex["input"]["paragraphs"]), gold_indices)
            yield qid, template, condition, ex, indices


def run_reader(args, db, examples, ranks, cap, frozen_template):
    stage, reader = args.stage, args.model
    jobs = list(answer_jobs(stage, examples, ranks, frozen_template))
    existing = {(qid, template, condition) for qid, template, condition in db.execute(
        "SELECT qid, template, condition FROM answers WHERE stage=? AND model=? AND cap=?", (stage, reader, cap))}
    todo = [job for job in jobs if job[:3] not in existing]
    if not todo:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for reader generation and comparable latency measurements.")
    device = "cuda"
    model_id, revision = MODELS[reader]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    free, total = torch.cuda.mem_get_info()
    headroom = 2**30
    if free <= headroom:
        raise RuntimeError("The GPU has less than 1 GiB of free VRAM.")
    torch.cuda.set_per_process_memory_fraction((free - headroom) / total)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_set = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    generation = GenerationConfig(max_new_tokens=cap, do_sample=False, num_beams=1, use_cache=True,
                                  cache_implementation="dynamic", return_dict_in_generate=False,
                                  bos_token_id=model.generation_config.bos_token_id,
                                  eos_token_id=eos_ids, pad_token_id=pad_id)
    # Replace reader-specific sampling defaults with one controlled greedy configuration.
    model.generation_config = generation
    template_kwargs = {"enable_thinking": False} if reader == "qwen3" else {}
    context_limit = model.config.max_position_embeddings
    config = {
        "model": model_id, "revision": revision, "precision": str(dtype), "quantization": None,
        "official_chat_template": tokenizer.chat_template, "chat_kwargs": template_kwargs,
        "generation": generation.to_dict(),
        "context_limit": context_limit, "batch_size": 1,
        "attention": "sdpa", "vram_headroom_bytes": headroom, "warmup_questions_per_invocation": WARMUPS,
        "hardware": hardware(torch, device), "software": software(),
        "timing": "Synchronized wall time from chat rendering/tokenization to decoded completion at batch size one, "
                  "with KV cache enabled within each answer. Excludes model loading, scoring, and database I/O.",
    }
    pin_config(db, f"reader/{reader}/{cap}", config)

    def generate(ex, indices, template):
        synchronize(torch, device)
        start = perf_counter()
        messages = make_messages(ex["input"], indices, template)
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt", **template_kwargs)
        n_prompt = inputs["input_ids"].shape[-1]
        if n_prompt + cap > context_limit:
            raise ValueError(f"Prompt for {ex['qid']} exceeds {model_id}'s context limit; no text was truncated.")
        with torch.inference_mode():
            output = model.generate(**inputs.to(device), generation_config=generation)
        new_ids = output[0, n_prompt:].tolist()
        raw = tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        extracted = clean_answer(tokenizer.decode(new_ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False))
        synchronize(torch, device)
        seconds = perf_counter() - start
        hit_cap = len(new_ids) == cap and new_ids[-1] not in eos_set
        return n_prompt, new_ids, raw, extracted, seconds, hit_cap

    # Warm up separately so first-use CUDA overhead does not enter the saved latencies.
    _, template, _, ex, indices = todo[0]
    for _ in range(WARMUPS):
        generate(ex, indices, template)
    torch.cuda.reset_peak_memory_stats()

    for index, (qid, template, condition, ex, indices) in enumerate(todo, 1):
        n_prompt, new_ids, raw, extracted, seconds, hit_cap = generate(ex, indices, template)
        gold = set(ex["reference"]["gold_indices"])
        kept = len(gold.intersection(indices))
        selector = condition.split("@")[0]
        selection_seconds = ranks[qid, selector]["uncached_seconds"] if selector in SELECTORS else None
        em, f1 = answer_scores(extracted, ex["reference"]["answer"])
        payload = {
            "qid": qid, "model": reader, "template": template, "condition": condition, "question_type": ex["question_type"],
            "selected_indices": indices, "gold_kept": kept, "gold_count": len(gold), "gold_recall": kept / len(gold),
            "all_gold": kept == len(gold), "gold_title_matches": ex["audit"]["gold_title_matches"],
            "gold_text_matches": ex["audit"]["gold_text_matches"],
            "raw_generation": raw, "generated_token_ids": new_ids, "answer": extracted,
            "gold_answer": ex["reference"]["answer"], "em": em, "f1": f1, "prompt_tokens": n_prompt,
            "generated_tokens": len(new_ids), "hit_output_cap_without_eos": hit_cap,
            "insufficient_evidence": extracted.casefold() == "insufficient evidence",
            "answer_seconds": seconds, "uncached_selector_seconds": selection_seconds,
            "total_seconds": None if condition == "oracle" else (selection_seconds or 0.0) + seconds,
            "vram_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "vram_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        with db:
            db.execute("INSERT OR IGNORE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (stage, reader, cap, template, condition, qid, json.dumps(payload, allow_nan=False)))
        if index % 25 == 0 or index == len(todo):
            print(f"{stage}/{reader}: {index}/{len(todo)} new answers", flush=True)


def saved_answers(db, stage, model, cap):
    return [json.loads(payload) for (payload,) in db.execute(
        "SELECT payload FROM answers WHERE stage=? AND model=? AND cap=? ORDER BY template, condition, qid",
        (stage, model, cap))]


def metrics(rows):
    from statistics import mean

    return {
        "n": len(rows), "em": mean(r["em"] for r in rows) if rows else None,
        "f1": mean(r["f1"] for r in rows) if rows else None,
        "gold_recall": mean(r["gold_recall"] for r in rows) if rows else None,
        "all_gold_rate": mean(r["all_gold"] for r in rows) if rows else None,
        "mean_prompt_tokens": mean(r["prompt_tokens"] for r in rows) if rows else None,
        "abstentions": sum(r["insufficient_evidence"] for r in rows),
        "output_cap_hits": sum(r["hit_output_cap_without_eos"] for r in rows),
    }


def strata(rows):
    yield "all", rows
    for kind in ("text", "title"):
        for matches in (0, 1, 2):
            yield f"gold_{kind}_matches_{matches}", [r for r in rows if r[f"gold_{kind}_matches"] == matches]


def require_complete(db, stage, examples, cap, frozen_template=None):
    ranks = get_rankings(db, examples)
    expected = {item[:3] for item in answer_jobs(stage, examples, ranks, frozen_template)}
    combined = []
    for model in MODELS:
        rows = saved_answers(db, stage, model, cap)
        keys = {(row["qid"], row["template"], row["condition"]) for row in rows}
        if keys != expected:
            raise ValueError(f"{stage}/{model}/{cap} needs {len(expected)} answers, currently {len(keys & expected)} complete.")
        combined.extend(rows)
    return combined


def freeze(out, db, examples, cap):
    dev = require_complete(db, "dev", examples, cap)
    cap_hits = sum(row["hit_output_cap_without_eos"] for row in dev)
    results = []
    for template in TEMPLATES:
        matched = [r for r in dev if r["template"] == template]
        for subset, rows in strata(matched):
            results.append({"template": template, "model": "all", "condition": "all", "subset": subset, **metrics(rows)})
        for model in MODELS:
            for selector in SELECTORS:
                group = [r for r in matched if r["model"] == model and r["condition"] == f"{selector}@2"]
                for subset, rows in strata(group):
                    results.append({"template": template, "model": model, "condition": f"{selector}@2",
                                    "subset": subset, **metrics(rows)})
    eligible = [r for r in results if r["model"] == "all" and r["subset"] == "all"]
    # With 100 answers in each configuration, pooled EM and F1 weight all four configurations equally.
    best = max(eligible, key=lambda r: (r["em"], r["f1"]))
    frozen = {"template": best["template"], "instruction": TEMPLATES[best["template"]],
              "abstention": ABSTENTION, "max_new_tokens": cap, "dev_em": best["em"], "dev_f1": best["f1"],
              "dev_answers_at_cap_without_eos": cap_hits,
              "tie_break": "Highest mean dev EM, then mean F1, then declared template order."}
    pin_config(db, "frozen", frozen)
    write_csv(out / "dev_template_results.csv", results)
    write_json(out / "frozen.json", frozen)
    print(f"Frozen: {best['template']} at {cap} generated tokens. Dev EM {best['em']:.3%}; {cap_hits} cap hits.")


def primary_statistics(groups):
    import numpy as np
    from scipy.stats import binomtest

    comparisons = []
    for model in MODELS:
        full = {r["qid"]: r["em"] for r in groups[model, "full"]}
        for selector in SELECTORS:
            condition = f"{selector}@2"
            differences = np.array([r["em"] - full[r["qid"]] for r in groups[model, condition]])
            rng = np.random.default_rng(42)
            resamples = rng.integers(len(differences), size=(10_000, len(differences)))
            low, high = np.quantile(differences[resamples].mean(axis=1), [.025, .975])
            fixed, broken = int((differences == 1).sum()), int((differences == -1).sum())
            p_value = binomtest(fixed, fixed + broken, .5).pvalue if fixed + broken else 1.0
            comparisons.append({"model": model, "condition": condition, "n": len(differences),
                                "em_difference": float(differences.mean()), "ci_95_low": float(low),
                                "ci_95_high": float(high), "fixed": fixed, "broken": broken,
                                "p_exact_mcnemar": p_value})
    running = 0.0
    for rank, row in enumerate(sorted(comparisons, key=lambda r: r["p_exact_mcnemar"])):
        running = max(running, min(1.0, (len(comparisons) - rank) * row["p_exact_mcnemar"]))
        row["p_holm"] = running
        row["significant_at_0_05_holm"] = running < .05
    return comparisons


def summarize_condition(model, condition, subset, rows, full):
    from statistics import mean

    result = {"model": model, "condition": condition, "subset": subset, **metrics(rows)}
    refs = [full[r["qid"]] for r in rows]
    prompt_total = sum(r["prompt_tokens"] for r in rows)
    full_prompt_total = sum(r["prompt_tokens"] for r in refs)
    correct_count = sum(r["em"] for r in rows)
    full_correct = sum(r["em"] for r in refs)
    fixed = sum(not ref["em"] and row["em"] for row, ref in zip(rows, refs))
    broken = sum(ref["em"] and not row["em"] for row, ref in zip(rows, refs))
    result.update(
        token_reduction=1 - prompt_total / full_prompt_total if refs else None,
        answer_preservation=(full_correct - broken) / full_correct if full_correct else None,
        fixed=fixed, broken=broken, prompt_tokens_per_correct_answer=prompt_total / correct_count if correct_count else None,
    )
    for field in ("uncached_selector_seconds", "answer_seconds", "total_seconds"):
        measured = [r[field] for r in rows if r[field] is not None]
        result[f"mean_{field}"] = mean(measured) if measured else None
    return result


def plot_test(out, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = {(r["model"], r["condition"]): r for r in summaries if r["subset"] == "all"}
    colors = {"dense": "#2563a6", "cross": "#bc552e"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True, layout="constrained")
    for model, ax in zip(MODELS, axes):
        for selector, label in (("dense", "Dense bi-encoder"), ("cross", "Fine-tuned cross-encoder")):
            rows = [result[model, f"{selector}@{k}"] for k in range(1, 6)]
            xs, ys = [100 * r["token_reduction"] for r in rows], [100 * r["em"] for r in rows]
            ax.plot(xs, ys, color=colors[selector], marker="o", label=label)
            for i, (x, y) in enumerate(zip(xs, ys), 1):
                ax.annotate(str(i), (x, y), xytext=(0, 8 if selector == "dense" else -13),
                            textcoords="offset points", ha="center", color=colors[selector])
        for condition, marker, label in (("full", "s", "Full context"), ("oracle", "D", "Oracle")):
            row = result[model, condition]
            ax.scatter(100 * row["token_reduction"], 100 * row["em"], marker=marker, color="#303846", label=label)
        ax.set_title(MODELS[model][0].split("/")[1])
        ax.set_xlabel("Prompt token reduction (%)")
        ax.grid(alpha=0.23)
    axes[0].set_ylabel("HotpotQA answer EM (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Answer accuracy and token reduction on 300 HotpotQA questions")
    fig.savefig(out / "accuracy_vs_token_reduction.png", dpi=180)
    plt.close(fig)


def report(out, db, examples, ranks, frozen, audit_summary):
    test = require_complete(db, "test", examples, frozen["max_new_tokens"], frozen["template"])
    groups = {(m, c): [r for r in test if r["model"] == m and r["condition"] == c] for m in MODELS for c in CONDITIONS}
    aggregates = []
    for (model, condition), rows in groups.items():
        full = {r["qid"]: r for r in groups[model, "full"]}
        for subset, subset_rows in strata(rows):
            aggregates.append(summarize_condition(model, condition, subset, subset_rows, full))
        for row in rows:
            ref = full[row["qid"]]
            row.update(full_em=ref["em"], full_prompt_tokens=ref["prompt_tokens"],
                       token_reduction=1 - row["prompt_tokens"] / ref["prompt_tokens"],
                       fixed=int(not ref["em"] and row["em"]), broken=int(ref["em"] and not row["em"]))
    comparisons = primary_statistics(groups)
    write_csv(out / "test_answers.csv", test)
    write_csv(out / "test_aggregate.csv", aggregates)
    write_csv(out / "primary_comparisons.csv", comparisons)
    plot_test(out, aggregates)

    metadata = {key: json.loads(val) for key, val in db.execute("SELECT * FROM metadata")}
    by_condition = {(r["model"], r["condition"], r["subset"]): r for r in aggregates}
    untouched = audit_summary["test_sample"]["gold_text_matches"]["0"]
    truncation = {}
    for name in SELECTORS:
        rows = [ranks[ex["qid"], name] for ex in examples]
        n = sum(len(r["truncated_paragraph_indices"]) for r in rows)
        truncation[name] = {"max_input_tokens": metadata[f"selector/{name}"]["max_input_tokens"],
                            "truncated_paragraphs": n, "candidates": 10 * len(rows),
                            "truncated_queries": sum(r["question_truncated_if_separately_encoded"] for r in rows),
                            "cached_ranking_disagreements": sum(r["cached_ranking_differs"] is True for r in rows)}
    split_counts = {
        "test_total": len(examples), "unseen_gold_text_sample_size": untouched, "test_truncation": truncation,
        "timing": "Total latency is a serial estimate using the uncached selector plus reader time. "
                  "Full context has reader time only. Oracle has no deployable selector time.",
    }
    write_json(out / "test_diagnostics.json", split_counts)
    lines = [
        "# HotpotQA Qwen evaluation\n",
        f"Both readers used the **{frozen['template']}** instruction selected on dev with a "
        f"{frozen['max_new_tokens']}-token output cap. Results below use the same 300 questions per condition. "
        f"The zero-overlap subset has {untouched} questions with neither gold paragraph text seen in project training.\n",
        "| Reader | Selector (k=2) | All gold | EM | Δ EM vs full (95% CI) | Holm p | Tokens saved | EM on unseen text |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        m, c = row["model"], row["condition"]
        all_rows = by_condition[m, c, "all"]
        unseen = by_condition[m, c, "gold_text_matches_0"]
        unseen_em = f"{unseen['em']:.1%}" if unseen["em"] is not None else "—"
        ci = f"{100 * row['em_difference']:+.1f} pp ({100 * row['ci_95_low']:+.1f}, {100 * row['ci_95_high']:+.1f})"
        lines.append(f"| {MODELS[m][0].split('/')[1]} | {c.split('@')[0]} | {all_rows['all_gold_rate']:.1%} | "
                     f"{all_rows['em']:.1%} | {ci} | {row['p_holm']:.3g} | {all_rows['token_reduction']:.1%} | {unseen_em} |")
    lines.append("")
    for m in MODELS:
        d, c = (by_condition[m, f"{s}@2", "all"] for s in SELECTORS)
        direction = "higher" if c["em"] > d["em"] else "lower" if c["em"] < d["em"] else "identical"
        lines.append(f"- **{MODELS[m][0].split('/')[1]}**. At k=2, cross-encoder EM was {direction} "
                     f"than dense EM ({c['em']:.1%} vs {d['em']:.1%}). The cross-encoder kept all annotated "
                     f"paragraphs in {c['all_gold_rate']:.1%} of questions and the dense selector in "
                     f"{d['all_gold_rate']:.1%}. This selector comparison is descriptive.")
    lines.extend([
        "",
        "The selection comparison combines architecture and training differences. The dense checkpoint was "
        "pretrained for matching queries with passages and received no project training. The cross-encoder was "
        "fine-tuned on project HotpotQA contexts. Both scored the same supplied candidates.",
        "",
        "The project previously used this test split and tuned an earlier prompt with the historical 300-question "
        "sample. The overlap audit covers project training only. It cannot rule out overlap with the models’ "
        "original training data or detect near-duplicate text. Both readers are in the Qwen family. The paired "
        "95% bootstrap intervals are pointwise. Significance claims use exact McNemar p-values adjusted with Holm "
        "across the four primary comparisons. A nonsignificant result does not establish equivalence.",
        "",
        "The full k curves and overlap strata are in test_aggregate.csv. Answer-cap hits are retained and counted. "
        "The pilot used greedy decoding so both readers have the same decoding policy. No k was selected or recommended "
        "using test data.",
    ])
    (out / "interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(test)} test answers, aggregates, statistics and plot to {out}.")


def export_rankings(out, db, examples):
    rows_by_qid = {ex["qid"]: ex for ex in examples}
    rows = []
    for qid, selector, payload in db.execute("SELECT qid, selector, payload FROM rankings ORDER BY selector, qid"):
        if qid not in rows_by_qid:
            continue
        row = json.loads(payload)
        row["selected_indices_by_k"] = {k: sorted(row["ranking"][:k]) for k in range(1, 6)}
        rows.append(row)
    return rows


def report_selection(out, split, examples, rank_rows):
    from statistics import mean

    by_id = {ex["qid"]: ex for ex in examples}
    metrics_rows = []
    diagnostics = {}
    for selector in SELECTORS:
        ranked = [row for row in rank_rows if row["selector"] == selector and row["qid"] in by_id]
        if len(ranked) != len(examples):
            continue
        diagnostics[selector] = {
            "questions": len(ranked), "candidate_count": len(ranked) * 10,
            "truncated_paragraph_count": sum(len(row["truncated_paragraph_indices"]) for row in ranked),
            "truncated_question_count": sum(row["question_truncated_if_separately_encoded"] for row in ranked),
            "mean_uncached_seconds": mean(row["uncached_seconds"] for row in ranked),
            "cached_dense_rankings_changed": sum(row["cached_ranking_differs"] is True for row in ranked),
        }
        for k in range(1, 6):
            rows = []
            for row in ranked:
                ex = by_id[row["qid"]]
                gold = set(ex["reference"]["gold_indices"])
                hits = gold.intersection(row["ranking"][:k])
                rows.append({**ex["audit"], "recall": len(hits) / len(gold), "all_gold": hits == gold})
            for subset, group in strata(rows):
                metrics_rows.append({"selector": selector, "k": k, "subset": subset, "n": len(group),
                                     "gold_recall": mean(r["recall"] for r in group) if group else None,
                                     "all_gold_rate": mean(r["all_gold"] for r in group) if group else None})
    write_csv(out / f"{split}_selection.csv", metrics_rows)
    write_json(out / f"{split}_selector_diagnostics.json", diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/qwen_hotpot"))
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("prepare", help="Reconstruct split IDs and audit paragraph overlap.")
    rank = commands.add_parser("rank", help="Score and cache the ten supplied paragraphs.")
    rank.add_argument("--split", choices=["pilot", "dev", "test"], required=True)
    rank.add_argument("--selector", choices=[*SELECTORS, "both"], default="both")
    rank.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    rank.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    for stage in ("pilot", "dev", "test"):
        reader = commands.add_parser(stage, help=f"Generate {stage} answers for one model.")
        reader.add_argument("--model", choices=MODELS, required=True)
        if stage != "test":
            reader.add_argument("--max-new-tokens", type=int, default=PILOT_START_CAP)
    commands.add_parser("freeze", help="Select one instruction by mean dev EM then F1.").add_argument(
        "--max-new-tokens", type=int, default=PILOT_START_CAP)
    commands.add_parser("report", help="Write aggregate metrics, paired statistics and the accuracy plot.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    out = args.out
    prepared_path = out / "prepared.json"
    if args.stage == "prepare":
        if prepared_path.exists():
            parser.exit(message=f"{prepared_path} already exists. Reusing its fixed samples.\n")
        prepare(out)
        return
    if not prepared_path.exists():
        parser.error(f"Run prepare first. Missing {prepared_path}.")
    prepared = load_json(prepared_path)
    examples = prepared["examples"]
    pilot_ids = set(prepared["manifest"]["pilot_ids"])
    examples["pilot"] = [ex for ex in examples["dev"] if ex["qid"] in pilot_ids]
    try:
        with open_store(out) as db:
            pin_config(db, "prepared", {"sha256": file_hash(prepared_path), "code_sha256": code_hash()})
            if (args.stage in ("test", "report") or (args.stage == "rank" and args.split == "test")):
                frozen = load_json(out / "frozen.json")
                pin_config(db, "frozen", frozen)
            if args.stage == "rank":
                selectors = SELECTORS if args.selector == "both" else (args.selector,)
                cross = checkpoint_info(args.checkpoint) if "cross" in selectors else None
                for selector in selectors:
                    run_ranking(args, db, examples[args.split], selector, cross)
                rank_rows = export_rankings(out, db, examples["dev"] + examples["test"])
                write_json(out / "selector_scores.json", rank_rows)
                report_selection(out, args.split, examples[args.split], rank_rows)
            elif args.stage == "freeze":
                freeze(out, db, examples["dev"], args.max_new_tokens)
            elif args.stage == "report":
                report(out, db, examples["test"], get_rankings(db, examples["test"]), frozen,
                       load_json(out / "overlap_summary.json"))
            else:
                cap = frozen["max_new_tokens"] if args.stage == "test" else args.max_new_tokens
                if cap < PILOT_START_CAP:
                    raise ValueError(f"Use at least {PILOT_START_CAP} generated tokens.")
                if args.stage == "dev":
                    pilot = require_complete(db, "pilot", examples["pilot"], cap)
                    if any(r["hit_output_cap_without_eos"] for r in pilot):
                        print("The pilot had output-cap hits. Review its saved generations before freezing this cap.", flush=True)
                stage_examples = examples[args.stage]
                ranks = get_rankings(db, stage_examples)
                template = frozen["template"] if args.stage == "test" else None
                run_reader(args, db, stage_examples, ranks, cap, template)
                rows = saved_answers(db, args.stage, args.model, cap)
                write_csv(out / f"{args.stage}_{args.model}_{cap}.csv", rows)
                if args.stage == "pilot":
                    pilot_summary = {
                        **metrics(rows),
                        "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
                        "peak_allocated_gib": max(r["vram_peak_allocated_bytes"] for r in rows) / 2**30,
                        "peak_reserved_gib": max(r["vram_peak_reserved_bytes"] for r in rows) / 2**30,
                        "cap_hit_conditions": [f"{r['qid']}/{r['template']}/{r['condition']}"
                                               for r in rows if r["hit_output_cap_without_eos"]],
                    }
                    write_json(out / f"pilot_{args.model}_{cap}_summary.json", pilot_summary)
                    print(json.dumps(pilot_summary, indent=2))
            config = {key: json.loads(val) for key, val in db.execute("SELECT * FROM metadata")}
            write_json(out / "run_configuration.json", {**prepared["manifest"], "runtime": config})
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{args.stage}: {error}\n")


if __name__ == "__main__":
    main()
