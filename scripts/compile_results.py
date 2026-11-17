"""
Week 2 deliverable: compile LSTM vs ARIMA results into a single comparison
table + bar chart for Ch4.

Usage:
    python compile_results.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
APPLIANCES = ["fan", "geyser", "computer"]


def load(appliance, model):
    path = RESULTS_DIR / f"{appliance}_{model}_results.json"
    with open(path) as f:
        return json.load(f)


def main():
    rows = []
    for appliance in APPLIANCES:
        lstm = load(appliance, "lstm")
        arima = load(appliance, "arima")
        rows.append({
            "appliance": appliance,
            "lstm_rmse_w": lstm["metrics"]["rmse_watts"],
            "arima_rmse_w": arima["metrics"]["rmse_watts"],
            "lstm_mae_w": lstm["metrics"]["mae_watts"],
            "arima_mae_w": arima["metrics"]["mae_watts"],
            "lstm_r2": lstm["metrics"]["r2"],
            "arima_r2": arima["metrics"]["r2"],
            "lstm_nrmse_pct": lstm["metrics"].get("nrmse_pct"),
            "arima_nrmse_pct": arima["metrics"].get("nrmse_pct"),
            "lstm_r2_reliable": lstm["metrics"].get("r2_reliable"),
            "arima_r2_reliable": arima["metrics"].get("r2_reliable"),
            "lstm_epochs_run": lstm["epochs_run"],
        })

    df = pd.DataFrame(rows)
    df["rmse_improvement_pct"] = (
        (df["arima_rmse_w"] - df["lstm_rmse_w"]) / df["arima_rmse_w"] * 100
    )
    df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(df.to_string(index=False))

    # bar chart: RMSE comparison
    x = np.arange(len(APPLIANCES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, df["lstm_rmse_w"], width, label="LSTM")
    ax.bar(x + width / 2, df["arima_rmse_w"], width, label="ARIMA")
    ax.set_xticks(x)
    ax.set_xticklabels(APPLIANCES)
    ax.set_ylabel("RMSE (W)")
    ax.set_title("LSTM vs ARIMA — RMSE by Appliance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lstm_vs_arima_rmse.png", dpi=120)
    plt.close()

    print(f"\nSaved comparison_table.csv and lstm_vs_arima_rmse.png to {RESULTS_DIR}")

    print("\nIMPORTANT CAVEATS (read before reporting these numbers anywhere):")
    print("1. This is now REAL iAWE data (not synthetic), full 100-epoch budget "
          "with early stopping -- a real improvement over the Week 2 dev run.")
    print("2. LSTM and ARIMA are NOT evaluated on identical held-out samples: "
          "LSTM uses a session-shuffled window split (preprocess.py), ARIMA "
          "aggregates rolling-forecast chunks across multiple sessions "
          "independently (train_arima.py). RMSE magnitudes are not strictly "
          "apples-to-apples between the two models -- R2/NRMSE within each "
          "model are more trustworthy than a direct RMSE bar-chart comparison.")
    print("3. On this specific run, ARIMA is competitive with or ahead of the "
          "LSTM on geyser and computer channels -- this does NOT match the "
          "'LSTM crushes ARIMA on bursty loads' narrative assumed earlier. "
          "With only 8-step (2h) input windows and a few hundred training "
          "samples, the LSTM doesn't have enough context or data to exploit "
          "its long-memory advantage over ARIMA yet. Recommended next step: "
          "add time-of-day/day-of-week cyclical features to the LSTM input "
          "(ARIMA can't use exogenous features as easily) to give it a real "
          "edge, OR honestly report the nuanced result in Ch4 rather than "
          "overselling LSTM superiority that the current data doesn't support.")


if __name__ == "__main__":
    main()
