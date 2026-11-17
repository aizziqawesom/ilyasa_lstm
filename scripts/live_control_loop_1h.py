"""
Live control loop using the 1-hour-window models (warm-up about 1 hour
instead of 2).

Identical to live_control_loop.py in every way (same Blynk pins, same
control_logic.decide(), same fail-safes on the ESP32 side) except:
    - INPUT_STEPS = 4 (1 hour of 15-min buckets) instead of 8
    - loads models from models_1h/ and scalers from data/processed_1h/
    - keeps its own buffer in data/live_1h/ so it never mixes with the 2h loop

Train the models first (once):
    python train_lstm_1h.py --all

Then fill in BLYNK_TOKEN and CURRENT_APPLIANCE in live_control_loop.py
(this script reuses those settings) and run:
    python live_control_loop_1h.py           # continuous, every 15 min
    python live_control_loop_1h.py --once    # single cycle

Warm-up: 4 readings, taken 15 min apart, so the first prediction arrives
about 45-60 minutes after starting (plus up to 15 min for the ESP32's first
V5 average after power-up).
"""

from pathlib import Path

import live_control_loop as loop

BASE_DIR = Path(__file__).resolve().parent.parent

loop.INPUT_STEPS = 4
loop.MODELS_DIR = BASE_DIR / "models_1h"
loop.PROCESSED_DIR = BASE_DIR / "data" / "processed_1h"
loop.LIVE_DIR = BASE_DIR / "data" / "live_1h"
loop.LIVE_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    missing = [a for a in ("geyser", "fan") if not (loop.MODELS_DIR / f"lstm_{a}.keras").exists()]
    if missing:
        raise SystemExit(f"1h models not found for {missing}. Run: python train_lstm_1h.py --all")
    import blynk_chart
    blynk_chart.install()   # adds the V21 / V22 lines on the Blynk Power history chart
    loop.main()
