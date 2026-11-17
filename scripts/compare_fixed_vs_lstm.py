"""
Fixed threshold vs LSTM forecast: which one acts at the right time?

Offline test on the held-out iAWE test windows (1-hour models, models_1h/).
No hardware needed. Same idea as the live system, two ways of deciding:

  FIXED (normal relay system): act if the reading NOW is above the limit.
  LSTM  (this project)       : act if the LSTM's forecast for the next 30 min
                               is above the limit.
  BOTH                       : act if either of the above is true.

Truth = what really happened in the next 30 min (the test set's y values):
"high load coming" means the real power in the next 30 min goes above the limit.

For each method we count, on the same windows:
  caught        high load coming, and the method acted       (good)
  missed        high load coming, but the method did nothing (bad)
  false alarm   method acted, but no high load came          (bad: needless cut/warning)
  early catch   high load coming, reading NOW still below the limit,
                and the method acted anyway. A fixed threshold can never
                do this, by definition. This is the LSTM's whole point.
  stale alarm   reading NOW above the limit, but the load drops in the next
                30 min, and the method still acted (acting too late)

It is repeated at 3 limit levels, because the real system tightens the limit
near a TNB bracket / in ToU peak:
  x1.00 normal, x0.85 halfway to the bracket, x0.70 at the bracket.

Limits are in the training data's own watts (iAWE). The water-heater model is
trained on the AC channel, so it uses the AC-proxy limit 1900 W (the original
calibration; the live 1620 W is the same limit rescaled for the real 1500 W heater).

Usage (in scripts/):
    python compare_fixed_vs_lstm.py
Output: results_1h/fixed_vs_lstm.csv (+ printed table)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed_1h"
MODELS = BASE / "models_1h"
OUT = BASE / "results_1h" / "fixed_vs_lstm.csv"

# limits in training-data watts (see note above)
LIMITS_W = {"geyser": 1900.0, "fan": 85.0, "computer": 72.0}
NAMES = {"geyser": "water heater (AC proxy)", "fan": "fan (TV channel)", "computer": "laptop"}
MULTS = {"normal x1.00": 1.00, "near bracket x0.85": 0.85, "at bracket x0.70": 0.70}


def load(app):
    d = PROCESSED / app
    X, y = np.load(d / "X_test.npy"), np.load(d / "y_test.npy")
    with open(d / "scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    inv = lambda a: sc.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)  # noqa: E731
    model = keras.models.load_model(MODELS / f"lstm_{app}.keras")
    pred = inv(model.predict(X, verbose=0).reshape(len(X), -1))
    now_w = inv(X[:, -1, 0])
    future_w = inv(y.reshape(len(y), -1))
    return now_w, pred.max(axis=1), future_w.max(axis=1)


def score(acted, coming, below_now, above_now):
    caught = int((acted & coming).sum())
    events = int(coming.sum())
    fa = int((acted & ~coming).sum())
    n_acted = int(acted.sum())
    return {
        "caught": caught,
        "missed": events - caught,
        "false_alarms": fa,
        "early_catches": int((acted & coming & below_now).sum()),
        "stale_alarms": int((acted & ~coming & above_now).sum()),
        "recall_%": round(100 * caught / events, 1) if events else None,
        "precision_%": round(100 * caught / n_acted, 1) if n_acted else None,
    }


def main():
    rows = []
    for app, base_limit in LIMITS_W.items():
        now_w, lstm_w, future_w = load(app)
        for level, m in MULTS.items():
            lim = base_limit * m
            coming = future_w >= lim
            below, above = now_w < lim, now_w >= lim
            common = {"appliance": app, "limit_level": level, "limit_w": round(lim, 1),
                      "windows": len(now_w), "high_load_coming": int(coming.sum()),
                      "of_which_starts_from_below": int((coming & below).sum())}
            rows.append({**common, "method": "FIXED (reading now)", **score(above, coming, below, above)})
            rows.append({**common, "method": "LSTM (30-min forecast)",
                         **score(lstm_w >= lim, coming, below, above)})
            rows.append({**common, "method": "BOTH (now OR forecast)",
                         **score(above | (lstm_w >= lim), coming, below, above)})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    for app in LIMITS_W:
        print(f"\n=== {NAMES[app]} ===")
        print(df[df.appliance == app].drop(columns=["appliance"]).to_string(index=False))
    print(f"\nSaved {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
