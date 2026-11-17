"""
Week 2 deliverable: per-appliance ARIMA baseline, for the LSTM-vs-ARIMA
comparison narrative in Ch4 (fan: close; geyser: ARIMA fails on bursts;
laptop: ARIMA struggles with multi-modal distribution).

Uses the same resampled_15min.csv series produced by preprocess.py (Week 1),
NOT the windowed/scaled arrays -- ARIMA works on the raw (unscaled) time
series directly and needs its own train/test split aligned to the same
chronological cutoff used for the LSTM, so the two are comparable.

Usage:
    python train_arima.py --appliance geyser
    python train_arima.py --all
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
FORECAST_HORIZON = 2  # match LSTM's 2-step (30min) output for a fair comparison -- revised Week 3, see preprocess.py

# simple default order per appliance -- documented as a starting point;
# a proper grid search (auto_arima) is a nice-to-have if time allows in Week 3
DEFAULT_ORDERS = {
    "fan": (2, 1, 2),
    "geyser": (1, 1, 1),
    "computer": (2, 1, 2),
}


MIN_SESSION_LEN = 20  # need enough points for a meaningful ARIMA fit + holdout


def load_sessions(appliance: str) -> list:
    """
    Splits the full series into contiguous non-NaN sessions (same definition
    as preprocess.py's session-aware windowing) and returns ALL sessions
    long enough to fit+evaluate ARIMA on, not just the single longest one.

    Revised Week 3 (twice): first version used interpolate+dropna() on the
    whole series, which spliced disconnected recording sessions together as
    if consecutive -- invalid for ARIMA's autoregressive assumption. Second
    version used only the single longest session, but that session can
    itself be an unrepresentative near-idle stretch (this happened on both
    geyser and fan -- R2 blew up to -80 and -1.6 respectively because that
    one session had almost no variance). Aggregating rolling-forecast
    results across MANY sessions -- mirroring how the LSTM is evaluated
    across many shuffled sessions in preprocess.py -- gives a fair,
    representative comparison instead of a lucky/unlucky single sample.
    """
    path = PROCESSED_DIR / appliance / "resampled_15min.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    raw = df["power_w"]
    is_na = raw.isna()
    change = is_na != is_na.shift()
    session_id = change.cumsum()

    sessions = []
    for sid, grp in raw.groupby(session_id):
        if is_na[grp.index].iloc[0]:
            continue
        if len(grp) >= MIN_SESSION_LEN:
            sessions.append(grp)
    print(f"    found {len(sessions)} sessions with >= {MIN_SESSION_LEN} buckets "
          f"(total {sum(len(s) for s in sessions)} buckets)")
    return sessions


def rolling_forecast_eval_one_session(series: pd.Series, order, max_chunks_per_session=15):
    n = len(series)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    if val_end < 5 or n - val_end < FORECAST_HORIZON:
        return None, None  # session too small to hold out anything meaningful

    test_vals = series.iloc[val_end:].values
    history = list(series.iloc[:val_end].values)
    preds, actuals = [], []

    i, step = 0, 0
    while i + FORECAST_HORIZON <= len(test_vals) and step < max_chunks_per_session:
        try:
            fit = ARIMA(history, order=order).fit()
            forecast = fit.forecast(steps=FORECAST_HORIZON)
        except Exception:
            forecast = np.full(FORECAST_HORIZON, history[-1])
        actual_chunk = test_vals[i:i + FORECAST_HORIZON]
        preds.append(forecast)
        actuals.append(actual_chunk)
        history.extend(actual_chunk.tolist())
        i += FORECAST_HORIZON
        step += 1

    if not preds:
        return None, None
    return np.array(preds), np.array(actuals)


def rolling_forecast_eval(sessions: list, order, appliance: str, max_sessions=20):
    """Aggregates rolling-forecast results across multiple sessions (capped
    for sandbox runtime) instead of relying on a single session's luck."""
    all_preds, all_actuals = [], []
    n_used = 0
    for series in sessions[:max_sessions]:
        preds, actuals = rolling_forecast_eval_one_session(series, order)
        if preds is not None:
            all_preds.append(preds)
            all_actuals.append(actuals)
            n_used += 1
    print(f"    aggregated rolling-forecast results from {n_used}/{min(len(sessions), max_sessions)} sessions evaluated")
    if not all_preds:
        raise RuntimeError(f"No sessions produced usable ARIMA forecasts for {appliance}")
    return np.concatenate(all_preds), np.concatenate(all_actuals)


def evaluate(preds: np.ndarray, actuals: np.ndarray):
    rmse = float(np.sqrt(mean_squared_error(actuals.ravel(), preds.ravel())))
    mae = float(mean_absolute_error(actuals.ravel(), preds.ravel()))
    r2 = float(r2_score(actuals.ravel(), preds.ravel()))
    mean_abs_actual = float(np.mean(np.abs(actuals)))
    nrmse_pct = float(rmse / mean_abs_actual * 100) if mean_abs_actual > 1e-6 else float("nan")
    test_std = float(np.std(actuals))
    rng = float(np.max(actuals) - np.min(actuals))
    low_variance_warning = test_std < 0.01 * (rng + 1e-9) if rng > 1 else test_std < 1.0
    return {
        "rmse_watts": rmse,
        "mae_watts": mae,
        "r2": r2,
        "nrmse_pct": nrmse_pct,
        "test_std_watts": test_std,
        "r2_reliable": not low_variance_warning,
    }


def plot_predictions(preds, actuals, appliance):
    plt.figure(figsize=(10, 4))
    plt.plot(actuals.ravel(), label="Actual", linewidth=1)
    plt.plot(preds.ravel(), label="ARIMA Predicted", linewidth=1, alpha=0.8)
    plt.title(f"ARIMA Prediction vs Actual — {appliance} (test set, rolling forecast)")
    plt.xlabel("15-min step (chunked)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"{appliance}_arima_prediction_vs_actual.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def run_appliance(appliance: str):
    print(f"\n=== ARIMA baseline: {appliance} ===")
    sessions = load_sessions(appliance)
    order = DEFAULT_ORDERS[appliance]
    print(f"  order={order}")

    preds, actuals = rolling_forecast_eval(sessions, order, appliance)
    metrics = evaluate(preds, actuals)
    print(f"  TEST metrics (Watts): RMSE={metrics['rmse_watts']:.2f} "
          f"MAE={metrics['mae_watts']:.2f} R2={metrics['r2']:.4f} "
          f"(evaluated over {len(preds)} forecast chunks)")

    plot_path = plot_predictions(preds, actuals, appliance)

    result = {
        "appliance": appliance,
        "model": f"ARIMA{order}",
        "forecast_horizon_steps": FORECAST_HORIZON,
        "n_forecast_chunks": len(preds),
        "metrics": metrics,
        "prediction_plot": str(plot_path),
        "note": "Rolling-origin forecast, refit each chunk on expanding history, capped at 60 chunks for runtime.",
    }
    with open(RESULTS_DIR / f"{appliance}_arima_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved -> {appliance}_arima_results.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliance", choices=["fan", "geyser", "computer"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    appliances = ["fan", "geyser", "computer"] if (args.all or not args.appliance) else [args.appliance]
    all_results = [run_appliance(a) for a in appliances]

    print("\n=== ARIMA Summary ===")
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['appliance']:10s} RMSE={m['rmse_watts']:8.2f}W  "
              f"MAE={m['mae_watts']:8.2f}W  R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
