# HotpotQA prompt compression with two Qwen readers

The evaluator runs separately from `relevance_classifier.ipynb` and `run_experiments.py`. Put `evaluate_qwen_hotpot.py` and `qwen_hotpot_data.py` in the repository root. The evaluator reads the historical results to verify split reconstruction and writes new results under `results/qwen_hotpot/`.

## Before running

Use the same CUDA-capable Python environment for the entire run. With a CUDA build of PyTorch installed, install the project requirements and add Sentence Transformers.

```bash
python -m pip install -r requirements.txt "transformers>=4.57,<6" "sentence-transformers>=5,<6"
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Get the complete `models/minilm-hotpotqa-best/` directory from the machine that ran `r5_4epochs`. It should include `best_run.json`, the model config, the tokenizer files, and fine-tuned weights. The repository records this checkpoint as best on dev and excludes its weights. Give `rank` the folder path with `--checkpoint` if it lives elsewhere. The evaluator checks the run, split settings, and files and fingerprints the checkpoint. It uses the checkpoint's 256-token limit and the original notebook's question–passage input format.

The archived `r6_qwen3_4b` run used `Qwen/Qwen3-4B-Instruct-2507` and the earlier `r5_2epochs` cross-encoder. Its saved answers are a separate historical result.

## Run the experiment

Execute the stages in order from the repo root. Each reader command runs one model and exits before the next model loads. Ranking can use CPU with `--device cpu`. If using that option, apply it consistently across ranking stages since timings are hardware-specific.

```bash
python evaluate_qwen_hotpot.py prepare
python evaluate_qwen_hotpot.py rank --split pilot
python evaluate_qwen_hotpot.py pilot --model qwen25
python evaluate_qwen_hotpot.py pilot --model qwen3
```

The pilot uses the first three sampled dev questions and the longest sampled dev context by character count, four questions in the reconstructed sample. It covers all six template/selector combinations at k=2, plus full and oracle contexts with the Direct template. Inspect the two `pilot_*_summary.json` files for peak allocated and reserved VRAM, prompt size, and output-cap hits. The two pilot CSVs contain the original generations, extracted answers, selected paragraph indices, and scores. The output cap starts at 64 tokens. If the cap cuts off otherwise valid answers, repeat both pilot runs with a larger `--max-new-tokens` and use the same value for both dev commands and the freeze step. Repetitions that run into the cap are recorded separately from EOS-terminated completions.

```bash
python evaluate_qwen_hotpot.py rank --split dev
python evaluate_qwen_hotpot.py dev --model qwen25
python evaluate_qwen_hotpot.py dev --model qwen3
python evaluate_qwen_hotpot.py freeze
```

The dev stage saves 1,200 answers. The freeze step requires all 1,200, picks a shared template and records its full wording and output cap. Selection uses the mean EM across the four equally sized model/selector combinations, with mean F1 and then declared template order as tiebreakers. A freeze is immutable within an output directory.

```bash
python evaluate_qwen_hotpot.py rank --split test
python evaluate_qwen_hotpot.py test --model qwen25
python evaluate_qwen_hotpot.py test --model qwen3
python evaluate_qwen_hotpot.py report
```

The test stage saves 7,200 answers, with 3,600 per reader. Each completed ranking and generation is committed to SQLite. Rerun an interrupted command to continue with the same settings, software and hardware. Global `--out PATH` goes before the stage name and selects an independent run directory. Keep a single command active against an output directory.

## What to check in the outputs

| File | Contents |
|---|---|
| `question_ids.json` and `prepared.json` | Reconstructed split IDs, the fixed dev and test samples, the pilot IDs, and the versioned dataset and prompt configuration |
| `paragraph_overlap.csv` and `overlap_summary.json` | Per-question and aggregate title and normalized text overlap with all project training contexts |
| `selector_scores.json` | Ten scores and ranked indices per selector and question, top-k indices in original context order, truncation, and uncached ranking time |
| `*_selection.csv` and `*_selector_diagnostics.json` | Selection recall and all-gold rates by k and overlap stratum, plus truncation counts |
| `run_configuration.json` and `frozen.json` | Model and tokenizer revisions, checkpoint hashes, exact chat and instruction templates, precision, hardware, software, and generation settings |
| `dev_template_results.csv` | Per-template EM and F1 across all four configurations and for each configuration |
| `test_answers.csv` and `test_aggregate.csv` | Raw and extracted answers, scores, prompt tokens, fixed and broken answers, latency, and overlap-stratified metrics |
| `primary_comparisons.csv` | Exact paired McNemar tests, Holm adjustment across four k=2 comparisons, and pointwise 95% paired bootstrap intervals for EM differences |
| `accuracy_vs_token_reduction.png` and `interpretation.md` | Complete k curves with full-context and oracle reference points, plus a short results interpretation |

The test sample retains 295 historical IDs. Five historical questions had fewer than 10 supplied paragraphs, so the evaluator takes the next five eligible IDs in the original test shuffle. The historical test questions were used to choose the earlier prompt. The new template choice uses dev only. The test overlap counts for 0, 1, and 2 gold paragraph texts seen in project training are 33, 102, and 165. The sample's title-overlap counts happen to be identical, though title and text counts differ across the full split.

The dense selector uses normalized vectors and dot products to compute cosine similarity. Its exact title-plus-body embeddings are cached across repeated passages. The cross-encoder scores all ten question–paragraph pairs jointly. The selectors are independent. The dense selector was pretrained for retrieval and the cross-encoder was fine-tuned on project examples, so training and architecture both differ. Both reader models belong to the Qwen family.

Latency is measured at batch size one with a warmup and GPU synchronization, excluding loading and dataset setup. Selector latency bypasses score and embedding caches. Reader latency covers prompt rendering, tokenization, generation, and decoding. Total latency is the sum of these independently measured stages for selected contexts. Full-context latency is reader latency. Oracle has no deployable end-to-end latency. Token reduction uses summed actual prompt tokens relative to the same model and questions in full context. Qwen3 uses its official chat template with thinking disabled and greedy decoding for experimental control. The answer-extraction rule removes tokenizer special tokens and trims whitespace. Additional generated prose stays in the scored answer.
