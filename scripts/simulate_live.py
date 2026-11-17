"""
Fill the dashboard's Live and Today tabs with SIMULATED data (no circuit needed).

Runs the real 1-hour LSTM and control_logic.decide() over made-up readings for
the last few hours, and writes the same files the live loops write:
    data/live_1h/decisions.csv        (Live tab)
    data/live_fast/readings.csv       (Today tab, "actual" line)

    python simulate_live.py            # then: streamlit run dashboard.py

These are scenarios, not sensor data: say so if you show them. Delete
data/live_1h/decisions.csv and data/live_fast/readings.csv to go back to real data.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import live_control_loop as loop
from control_logic import decide

BASE_DIR = Path(__file__).resolve().parent.parent
loop.INPUT_STEPS = 4
loop.MODELS_DIR = BASE_DIR / "models_1h"
loop.PROCESSED_DIR = BASE_DIR / "data" / "processed_1h"
loop.LIVE_DIR = BASE_DIR / "data" / "live_1h"
loop.LIVE_DIR.mkdir(parents=True, exist_ok=True)
FAST_DIR = BASE_DIR / "data" / "live_fast"
FAST_DIR.mkdir(parents=True, exist_ok=True)

STEPS = 24                       # 15-min points = last 6 hours
rng = np.random.default_rng(0)
# appliance: (typical W, kWh at start, kWh at end of the run)
SIM = {"fan": (50, 150, 199), "geyser": (1500, 150, 197), "lamp": (3, 150, 196)}

now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
readings = []
for app, (typ, k0, k1) in SIM.items():
    w = typ * (1 + rng.normal(0, 0.02, STEPS))
    for i in range(STEPS):
        ts = now - timedelta(minutes=15 * (STEPS - 1 - i))
        local = ts.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
        readings.append({"time": local.isoformat(sep=" "), "appliance": app, "power_w": round(w[i], 2)})
        if i < 3:
            continue
        window = w[i - 3:i + 1]
        kwh = k0 + (k1 - k0) * i / (STEPS - 1)
        d = decide(app, float(window[-1]), loop.predict_next(app, window), kwh)
        forecast = loop.predict_next(app, window)
        loop.log_decision(ts, app, window, float(window[-1]), forecast, kwh, d)
    print(f"{app}: {STEPS - 3} decisions written")

pd.DataFrame(readings).to_csv(FAST_DIR / "readings.csv", index=False)
print("Done. Now run: streamlit run dashboard.py")
