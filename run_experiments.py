"""Run relevance_classifier.ipynb once per run setting with papermill.

Each run writes its executed notebook to experiments/<run>.ipynb (saved after every cell, so it can be
opened while the run is going) and adds one row to results/experiments.csv (Section 10 of the notebook).
Runs already in results/experiments.csv are skipped, so the script can be restarted after an interruption.

    python run_experiments.py              # every run below, in order
    python run_experiments.py r2_bs16      # selected runs only
    python run_experiments.py --fresh      # new sweep: first move the old results and best model to backups

The notebook runs on the Jupyter kernel "prompt-optimizer-nlp" (see the README). Set NB_KERNEL to use
another one, e.g. NB_KERNEL=python3.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import papermill as pm

# Everything not listed keeps the default of the notebook's Section 0 configuration cell
RUNS = {
    "r1_baseline":        dict(TRAIN_BATCH_SIZE=32, LEARNING_RATE=2e-5, EPOCHS=3, MAX_LENGTH=256),
    "r2_bs16":            dict(TRAIN_BATCH_SIZE=16, LEARNING_RATE=2e-5, EPOCHS=3, MAX_LENGTH=256),
    "r3_bs64_lr4e-5":     dict(TRAIN_BATCH_SIZE=64, LEARNING_RATE=4e-5, EPOCHS=3, MAX_LENGTH=256),
    "r4_len384":          dict(TRAIN_BATCH_SIZE=32, LEARNING_RATE=2e-5, EPOCHS=3, MAX_LENGTH=384),
    "r5_4epochs":         dict(TRAIN_BATCH_SIZE=32, LEARNING_RATE=2e-5, EPOCHS=4, MAX_LENGTH=256),
    "r7_bs64_lr3e-5_3ep": dict(TRAIN_BATCH_SIZE=64, LEARNING_RATE=3e-5, EPOCHS=3, MAX_LENGTH=256, WARMUP_FRAC=0.15),
    # Second answering LLM, run last: it reuses the best MiniLM of the runs above (not retrained) and its
    # settings, so only the reader changes. Qwen3-4B needs more VRAM per token than SmolLM3, so it runs
    # under a hard 12 GB limit with smaller generation batches.
    "r6_qwen3_4b":        dict(TRAIN_MINILM=False, LLM_NAME="Qwen/Qwen3-4B-Instruct-2507", GPU_MAX_GB=12.0,
                               GEN_MAX_BATCH_TOKENS=4096),
}

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "relevance_classifier.ipynb"
KERNEL = os.environ.get("NB_KERNEL", "prompt-optimizer-nlp")
EXPERIMENTS_CSV = ROOT / "results" / "experiments.csv"
BEST_DIR = ROOT / "models" / "minilm-hotpotqa-best"


def finished_runs():
    return set(pd.read_csv(EXPERIMENTS_CSV)["run"]) if EXPERIMENTS_CSV.exists() else set()


def start_fresh():
    # Keep the old report and best model as backups instead of deleting them (both are git-ignored)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    for path in (EXPERIMENTS_CSV, BEST_DIR):
        if path.exists():
            backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
            path.rename(backup)
            print(f"moved {path.relative_to(ROOT)} to {backup.relative_to(ROOT)}")


def minutes(seconds):
    return f"{seconds / 60:.1f} min"


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP downloads; Xet transfers hung on this machine
    args = sys.argv[1:]
    if "--fresh" in args:
        args.remove("--fresh")
        start_fresh()
    (ROOT / "experiments").mkdir(exist_ok=True)
    names = args or list(RUNS)
    sweep_t0 = time.time()
    for i, name in enumerate(names, 1):
        if name in finished_runs():
            print(f"[{i}/{len(names)}] skip {name}: already in results/experiments.csv")
            continue
        run_t0 = time.time()
        print(f"[{i}/{len(names)}] {name} started at {datetime.now():%H:%M} | {RUNS[name]}", flush=True)
        # The bar shows the notebook's cells; every cell's output is in experiments/<run>.ipynb
        pm.execute_notebook(NOTEBOOK, ROOT / "experiments" / f"{name}.ipynb",
                            parameters={"RUN_NAME": name, **RUNS[name]},
                            kernel_name=KERNEL, cwd=str(ROOT), log_output=False, progress_bar=True)
        print(f"[{i}/{len(names)}] {name} done in {minutes(time.time() - run_t0)} | "
              f"sweep so far {minutes(time.time() - sweep_t0)}", flush=True)
    print(f"All runs done in {minutes(time.time() - sweep_t0)}")
