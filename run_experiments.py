"""Run relevance_classifier.ipynb once per hyper-parameter combination with papermill.

Each run writes its executed notebook to experiments/<run>.ipynb (saved after every cell, so it can be
opened while the run is going) and adds one row to results/experiments.csv (Section 10 of the notebook).
Runs already in results/experiments.csv are skipped, so the script can be restarted after an interruption.

    .venv\\Scripts\\python run_experiments.py              # every run below
    .venv\\Scripts\\python run_experiments.py r2_bs16      # selected runs only
"""
import os
import sys
from pathlib import Path

import pandas as pd
import papermill as pm

# Everything not listed keeps the default of the notebook's Section 0 configuration cell
RUNS = {
    "r1_baseline":    dict(TRAIN_BATCH_SIZE=32, LEARNING_RATE=2e-5, EPOCHS=1, MAX_LENGTH=256),
    "r2_bs16":        dict(TRAIN_BATCH_SIZE=16, LEARNING_RATE=2e-5, EPOCHS=1, MAX_LENGTH=256),
    "r3_bs64_lr4e-5": dict(TRAIN_BATCH_SIZE=64, LEARNING_RATE=4e-5, EPOCHS=1, MAX_LENGTH=256),
    "r4_len384":      dict(TRAIN_BATCH_SIZE=32, LEARNING_RATE=2e-5, EPOCHS=1, MAX_LENGTH=384),
    "r5_2epochs":     dict(TRAIN_BATCH_SIZE=32, LEARNING_RATE=2e-5, EPOCHS=2, MAX_LENGTH=256),
}

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "relevance_classifier.ipynb"
KERNEL = "prompt-optimizer-nlp"


def finished_runs():
    path = ROOT / "results" / "experiments.csv"
    return set(pd.read_csv(path)["run"]) if path.exists() else set()


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP downloads; Xet transfers hung on this machine
    (ROOT / "experiments").mkdir(exist_ok=True)
    for name in sys.argv[1:] or list(RUNS):
        if name in finished_runs():
            print(f"skip {name}: already in results/experiments.csv")
            continue
        print(f"=== {name}: {RUNS[name]}", flush=True)
        pm.execute_notebook(NOTEBOOK, ROOT / "experiments" / f"{name}.ipynb",
                            parameters={"RUN_NAME": name, **RUNS[name]},
                            kernel_name=KERNEL, cwd=str(ROOT), log_output=True, progress_bar=False)
        print(f"=== {name}: done", flush=True)
