"""Recover the project's HotpotQA splits and audit paragraph overlap."""

import csv
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile


DATASET = "hotpotqa/hotpot_qa"
DATA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SPLIT = {"SEED": 42, "SAMPLE_FRAC": 1.0, "TEST_FRAC": 0.20, "DEV_FRAC": 0.05, "TRAIN_FRAC": 1.0}
SPLIT_SIZES = {"train": 74366, "dev": 3915, "test": 19571}
MODELS = {
    "qwen25": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "qwen3": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
}
DENSE = ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "b207367332321f8e44f96e224ef15bc607f4dbf0")
TEMPLATES = {
    "direct": "Answer the question using only the supplied passages. Return only the shortest complete answer, without explanation.",
    "evidence": "Identify the evidence relevant to the question in the supplied passages. Answer using only that evidence. Return only the shortest complete answer, without explanation.",
    "multihop": "Combine information across the supplied passages when necessary to answer the question. Return only the shortest complete answer, without explanation.",
}
ABSTENTION = "If the passages do not provide enough information, return ‘Insufficient evidence’."
PROMPT_FORMAT = "{instruction}\n\nPassages:\n{passages}\n\nQuestion: {question}"
DEV_SEED = 20261119
TEST_SEED = 42
K_VALUES = tuple(range(1, 6))
SELECTORS = ("dense", "cross")
CONDITIONS = ("full", "oracle", *(f"{selector}@{k}" for selector in SELECTORS for k in K_VALUES))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        json.dump(value, temp, ensure_ascii=False, indent=2, allow_nan=False)
        temp.write("\n")
    os.replace(temp.name, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as temp:
        writer = csv.DictWriter(temp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp.name, path)


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checkpoint_info(folder):
    meta_path = folder / "best_run.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Copy the trained models/minilm-hotpotqa-best folder here: {folder} (missing best_run.json)")
    meta = load_json(meta_path)
    if any(meta.get(key) != value for key, value in SPLIT.items()):
        raise ValueError("Checkpoint split or training fraction differs from the audited project setup.")
    if not meta.get("TRAIN_MINILM") or not meta.get("run") or not isinstance(meta.get("MAX_LENGTH"), (int, float)):
        raise ValueError("The checkpoint metadata must identify the trained run and its MAX_LENGTH.")
    required = [folder / "config.json", folder / "tokenizer_config.json"]
    weights = sorted(folder.glob("model*.safetensors")) or sorted(folder.glob("pytorch_model*.bin"))
    tokenizer_files = [p for p in (folder / "tokenizer.json", folder / "vocab.txt") if p.is_file()]
    if not all(p.is_file() for p in required) or not weights or not tokenizer_files:
        raise FileNotFoundError(f"Missing model config, tokenizer files or fine-tuned weights in {folder}")
    with (Path(__file__).parent / "results" / "experiments.csv").open(encoding="utf-8-sig", newline="") as source:
        recorded_best = next(row["run"] for row in csv.DictReader(source) if row["best_model"].lower() == "true")
    if meta["run"] != recorded_best:
        raise ValueError(f"This repo records {recorded_best} as the best run. The supplied checkpoint is {meta['run']}.")
    return {"metadata": meta, "weights": [file.name for file in weights],
            "sha256": {p.name: file_hash(p) for p in sorted(folder.iterdir()) if p.is_file()}}


def paragraph_pairs(row):
    context = row["context"]
    return [(title, "".join(sentences)) for title, sentences in
            zip(context["title"], context["sentences"], strict=True)]


def selector_text(title, text):
    # Same title/body formatting as the project's training notebook.
    return f"{title}. {text}"


def normalized(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def text_hash(s):
    return hashlib.sha256(normalized(s).encode("utf-8")).digest()


def gold_positions(row, pairs):
    titles = set(row["supporting_facts"]["title"])
    found = {title for title, _ in pairs}
    return [i for i, (title, _) in enumerate(pairs) if title in titles], titles - found


def historical_ids(repo, split):
    name = "r5_4epochs_answers.csv" if split == "test" else "r5_4epochs_dev_answers.csv"
    with (repo / "results" / "runs" / name).open(newline="", encoding="utf-8-sig") as f:
        return [row["qid"] for row in csv.DictReader(f) if row["condition"] == "full"]


def prepare(out):
    from datasets import concatenate_datasets, load_dataset
    from importlib.metadata import version

    data = load_dataset(DATASET, "distractor", revision=DATA_REVISION)
    full = concatenate_datasets([data["train"], data["validation"]])
    sampled = full.shuffle(seed=SPLIT["SEED"]).select(range(int(len(full) * SPLIT["SAMPLE_FRAC"])))
    first = sampled.train_test_split(test_size=SPLIT["TEST_FRAC"], seed=SPLIT["SEED"])
    second = first["train"].train_test_split(test_size=SPLIT["DEV_FRAC"], seed=SPLIT["SEED"])
    splits = {"train": second["train"], "dev": second["test"], "test": first["test"]}
    ids = {split: list(rows["id"]) for split, rows in splits.items()}
    sets = {split: set(qids) for split, qids in ids.items()}
    if ({s: len(qids) for s, qids in ids.items()} != SPLIT_SIZES
            or any(len(ids[s]) != len(sets[s]) for s in ids)
            or len(set.union(*sets.values())) != len(full)):
        raise ValueError("Reconstructed splits have unexpected sizes, duplicates or overlapping IDs.")

    old_order = {s: splits[s].shuffle(seed=TEST_SEED) for s in ("dev", "test")}
    repo = Path(__file__).parent
    history = {s: historical_ids(repo, s) for s in old_order}
    for s, qids in history.items():
        if len(qids) != 300 or list(old_order[s]["id"][:300]) != qids:
            raise ValueError(f"Reconstructed {s} order disagrees with the notebook's historical 300 IDs.")

    train_titles, train_text = set(), set()
    for row in splits["train"].select_columns(["context"]):
        for title, text in paragraph_pairs(row):
            train_titles.add(normalized(title))
            train_text.add(text_hash(text))

    audit_rows = []
    audit_by_id = {}
    for split in ("dev", "test"):
        columns = ["id", "context", "supporting_facts"]
        for row in splits[split].select_columns(columns):
            pairs = paragraph_pairs(row)
            gold, missing = gold_positions(row, pairs)
            title_hits = [i for i in gold if normalized(pairs[i][0]) in train_titles]
            text_hits = [i for i in gold if text_hash(pairs[i][1]) in train_text]
            audit = {"split": split, "qid": row["id"], "candidate_count": len(pairs), "gold_count": len(gold),
                     "gold_title_matches": len(title_hits), "gold_text_matches": len(text_hits),
                     "missing_gold_titles": len(missing), "title_match_indices": title_hits,
                     "text_match_indices": text_hits}
            audit_rows.append(audit)
            audit_by_id[row["id"]] = audit

    def eligible(qid):
        audit = audit_by_id[qid]
        return audit["candidate_count"] == 10 and audit["gold_count"] > 0 and not audit["missing_gold_titles"]

    test_order = list(old_order["test"]["id"])
    test_ids = [qid for qid in test_order if eligible(qid)][:300]
    dev_eligible = [qid for qid in ids["dev"] if eligible(qid)]
    dev_ids = random.Random(DEV_SEED).sample(dev_eligible, 100)
    if len(test_ids) != 300:
        raise ValueError("There are fewer than 300 eligible test questions.")

    examples = {}
    for split, sample in (("dev", dev_ids), ("test", test_ids)):
        position = {qid: idx for idx, qid in enumerate(ids[split])}
        examples[split] = []
        for row in splits[split].select([position[qid] for qid in sample]):
            pairs = paragraph_pairs(row)
            gold, _ = gold_positions(row, pairs)
            examples[split].append({
                "qid": row["id"], "question_type": row["type"],
                "input": {"question": row["question"], "paragraphs": pairs},
                "reference": {"answer": row["answer"], "gold_indices": gold},
                "audit": audit_by_id[row["id"]],
            })

    pilot = examples["dev"][:3]
    longest_dev = max(examples["dev"], key=lambda ex: sum(len(t) + len(p) for t, p in ex["input"]["paragraphs"]))
    pilot_ids = list(dict.fromkeys([*(ex["qid"] for ex in pilot), longest_dev["qid"]]))

    def census(rows):
        result = {"n": len(rows)}
        for field in ("candidate_count", "gold_count", "gold_title_matches", "gold_text_matches", "missing_gold_titles"):
            counts = Counter(row[field] for row in rows)
            if "matches" in field:
                counts.update({0: 0, 1: 0, 2: 0})
            result[field] = dict(sorted(counts.items()))
        return result

    summary = {
        "normalization": "Unicode NFKC, casefold, collapse whitespace. SHA-256 of paragraph body without title.",
        "training_scope": "Every paragraph in every question of the reconstructed project training split.",
        "train_unique_titles": len(train_titles), "train_unique_text_hashes": len(train_text),
        **{s: census([row for row in audit_rows if row["split"] == s]) for s in ("dev", "test")},
        **{f"{s}_sample": census([ex["audit"] for ex in examples[s]]) for s in ("dev", "test")},
    }
    manifest = {
        "dataset": DATASET, "revision": DATA_REVISION, "configuration": "distractor",
        "split_arguments": SPLIT, "split_sizes": SPLIT_SIZES, "id_intersections": 0,
        "data_software": {p: version(p) for p in ("datasets", "numpy", "pyarrow")},
        "previous_use": "Historical 300 test IDs informed prompt tuning, and the full project test split was scored.",
        "test_policy": "Keep historical seeded order and fill from its continuation only if an old ID is ineligible.",
        "test_seed": TEST_SEED, "dev_seed": DEV_SEED, "pilot_ids": pilot_ids,
        "historical_test_replacements": [qid for qid in history["test"] if qid not in test_ids],
        "eligibility": "Exactly 10 supplied passages, at least one gold paragraph and all annotated titles present.",
        "prompts": TEMPLATES, "abstention": ABSTENTION, "format": PROMPT_FORMAT, "passage_separator": "\n\n",
        "models": MODELS, "dense": DENSE,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paragraph_overlap.csv", audit_rows)
    write_json(out / "overlap_summary.json", summary)
    write_json(out / "question_ids.json", {**ids, "dev_sample": dev_ids, "test_sample": test_ids,
                                           "pilot": pilot_ids, "historical_test_sample": history["test"]})
    write_json(out / "prepared.json", {"manifest": manifest, "examples": examples})
    print(f"Prepared {len(dev_ids)} dev and {len(test_ids)} test questions. Audit in {out / 'overlap_summary.json'}.")


def select_indices(condition, rankings, count, oracle_indices=None):
    if condition == "full":
        return list(range(count))
    if condition == "oracle":
        return sorted(oracle_indices)
    selector, k = condition.split("@")
    return sorted(rankings[selector][:int(k)])


def make_messages(inputs, indices, template):
    passages = "\n\n".join(selector_text(*inputs["paragraphs"][i]) for i in indices)
    instruction = f"{TEMPLATES[template]} {ABSTENTION}"
    return [{"role": "user", "content": PROMPT_FORMAT.format(
        instruction=instruction, passages=passages, question=inputs["question"])}]


def clean_answer(raw):
    # Keep the whole completion, including accidental explanations, to avoid rescuing answers after the fact.
    return raw.strip()


def answer_scores(prediction, gold):
    import string

    def normalize_answer(answer):
        answer = answer.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", answer).split())

    pred, gold = normalize_answer(prediction), normalize_answer(gold)
    em = int(pred == gold)
    if pred != gold and ({pred, gold} & {"yes", "no", "noanswer"}):
        return em, 0.0
    pred_tokens, gold_tokens = pred.split(), gold.split()
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not common:
        return em, 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return em, 2 * precision * recall / (precision + recall)
