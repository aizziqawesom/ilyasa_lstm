"""
Week 3 deliverable: end-to-end validation. Loads each trained LSTM model,
runs it against its own real test-set windows, and feeds every prediction
through control_logic.decide() -- demonstrating the FULL pipeline
(sensor reading -> LSTM forecast -> control decision) on real data, not
just the model in isolation. This is the evidence for Ch4 that the control
module actually responds sensibly to real predicted trajectories.

Also simulates a rising monthly cumulative kWh (starting near a TNB tier
boundary) across the test sequence, since the real test windows don't carry
a "cumulative usage" value of their own -- this lets the dynamic threshold
behaviour be exercised and visible in the output log.

Usage:
    python run_control_simulation.py --appliance geyser
    python run_control_simulation.py --all
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from control_logic import decide, Action

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Starting cumulative monthly kWh per appliance's simulated household --
# chosen close to a tier boundary (200kWh) so the dynamic-threshold
# behaviour is actually visible in the demo log, not just sitting flat at 1.0x.
SIM_START_CUM_KWH = 178.0
SIM_KWH_INCREMENT_PER_STEP = 1.1  # sized so a 20-step demo sweeps across the full
                                    # dynamic-threshold tightening margin (180-200kWh),
                                    # showing the transition from normal to tightened
                                    # sensitivity rather than sitting at one extreme


def run_appliance(appliance: str, n_examples: int = 20):
    print(f"\n=== Control simulation: {appliance} ===")
    d = PROCESSED_DIR / appliance
    X_test = np.load(d / "X_test.npy")
    y_test = np.load(d / "y_test.npy")
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = keras.models.load_model(MODELS_DIR / f"lstm_{appliance}.keras")

    n = min(n_examples, len(X_test))
    idx = np.linspace(0, len(X_test) - 1, n).astype(int)  # spread across the test set

    y_pred_scaled = model.predict(X_test[idx], verbose=0)
    y_pred_w = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    current_w = scaler.inverse_transform(X_test[idx][:, -1, 0].reshape(-1, 1)).ravel()  # last observed input step

    cum_kwh = SIM_START_CUM_KWH
    rows = []
    for i in range(n):
        pred_next_30min_w = float(y_pred_w[i, 0])  # first forecasted 15-min step
        cur_w = float(current_w[i])
        decision = decide(
            appliance=appliance,
            current_power_w=cur_w,
            predicted_power_w=pred_next_30min_w,
            cumulative_kwh_this_month=cum_kwh,
        )
        rows.append({
            "step": i,
            "current_w": round(cur_w, 1),
            "predicted_w": round(pred_next_30min_w, 1),
            "cumulative_kwh": round(cum_kwh, 1),
            "threshold_w": round(decision.threshold_w, 1),
            "action": decision.action.value,
            "alert_tier": decision.alert_tier.value,
            "reason": decision.reason,
        })
        cum_kwh += SIM_KWH_INCREMENT_PER_STEP

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"{appliance}_control_simulation.csv"
    df.to_csv(out_path, index=False)

    action_counts = df["action"].value_counts().to_dict()
    print(f"  ran {n} control cycles | action distribution: {action_counts}")
    print(df[["step", "current_w", "predicted_w", "cumulative_kwh", "threshold_w", "action", "alert_tier"]].to_string(index=False))
    print(f"  saved -> {out_path.name}")

    return {"appliance": appliance, "n_cycles": n, "action_counts": action_counts, "csv": str(out_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliance", choices=["fan", "geyser", "computer"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    appliances = ["fan", "geyser", "computer"] if (args.all or not args.appliance) else [args.appliance]
    summary = [run_appliance(a) for a in appliances]

    with open(RESULTS_DIR / "control_simulation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved control_simulation_summary.json to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
