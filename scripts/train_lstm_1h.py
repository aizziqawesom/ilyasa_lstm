"""
Train a 1-hour-window variant of the LSTM models, so the live loop only needs
1 hour of warm-up instead of 2.

Why a separate model is needed: the original models (train_lstm.py) take
INPUT_STEPS = 8 x 15-min buckets = 2 hours of history as input. A model can
only be fed the input length it was trained on, so the live loop cannot start
predicting until it has collected 2 hours of real readings. To cut the
warm-up to 1 hour, the model itself has to be retrained with
INPUT_STEPS = 4 (4 x 15 min = 1 hour). Everything else is identical: same
iAWE data and appliance mapping, same session-aware windowing and split,
same architecture (2 x LSTM(64), dropout 0.2, Adam, MSE), same 30-min
(2-step) forecast.

Trade-off: the model sees half as much history, so accuracy may drop. The
script prints the 1h metrics next to the original 2h metrics so the
difference is measured, not assumed. Report whichever you use honestly in Ch4.

Outputs go to separate folders so the original 2h models are untouched:
    data/processed_1h/<appliance>/   (windows + scaler.pkl)
    models_1h/lstm_<appliance>.keras
    results_1h/<appliance>_lstm_results.json (+ plots)

Usage:
    python train_lstm_1h.py --all
    python train_lstm_1h.py --appliance geyser
"""

import argparse
import json
from pathlib import Path

import preprocess
import train_lstm

INPUT_STEPS_1H = 4  # 4 x 15 min = 1 hour

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_1H = BASE_DIR / "data" / "processed_1h"
MODELS_1H = BASE_DIR / "models_1h"
RESULTS_1H = BASE_DIR / "results_1h"
for d in (PROCESSED_1H, MODELS_1H, RESULTS_1H):
    d.mkdir(parents=True, exist_ok=True)

# Point both modules at the 1h settings and folders. Their functions read
# these module-level values at call time, so no other code changes needed.
preprocess.INPUT_STEPS = INPUT_STEPS_1H
preprocess.PROCESSED_DIR = PROCESSED_1H
train_lstm.INPUT_STEPS = INPUT_STEPS_1H
train_lstm.PROCESSED_DIR = PROCESSED_1H
train_lstm.MODELS_DIR = MODELS_1H
train_lstm.RESULTS_DIR = RESULTS_1H

APPLIANCES = ["fan", "geyser", "computer"]


def run(appliance: str, epochs: int, patience: int):
    print(f"\n=== {appliance}: 1-hour input window ({INPUT_STEPS_1H} x 15 min) ===")
    preprocess.process_appliance(appliance)
    train_lstm.train_appliance(appliance, epochs=epochs, patience=patience)

    new = json.loads((RESULTS_1H / f"{appliance}_lstm_results.json").read_text())
    old_path = BASE_DIR / "results" / f"{appliance}_lstm_results.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else None

    def fmt(r):
        m = r.get("metrics", r)
        return (f"RMSE={m.get('rmse_watts', float('nan')):.1f}W  "
                f"MAE={m.get('mae_watts', float('nan')):.1f}W  R2={m.get('r2', float('nan')):.3f}")

    print(f"  1h window : {fmt(new)}")
    if old:
        print(f"  2h window : {fmt(old)}  (original model, for comparison)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--appliance", choices=APPLIANCES)
    p.add_argument("--all", action="store_true")
    p.add_argument("--epochs", type=int, default=train_lstm.EPOCHS)
    p.add_argument("--patience", type=int, default=train_lstm.PATIENCE)
    a = p.parse_args()
    targets = APPLIANCES if a.all else [a.appliance] if a.appliance else None
    if not targets:
        p.error("use --all or --appliance")
    for ap in targets:
        run(ap, a.epochs, a.patience)


if __name__ == "__main__":
    main()
