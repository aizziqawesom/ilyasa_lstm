"""
FAST TEST loop - first decision after about 2 minutes, then one every 30 s.

Uses REAL live readings from your PZEM (via Blynk V0), the real 1-hour LSTM
(models_1h/) and the real control_logic.decide(). The only difference from
live_control_loop_1h.py is timing: it treats each 30-second reading as one
model "step" instead of one 15-minute bucket.

IMPORTANT, say this if you show it: the model was trained on 15-minute steps,
so with 30-second steps the "forecast" is not a real 30-minute forecast. This
mode tests that the whole chain works live (PZEM -> ESP32 -> Blynk -> Python
-> LSTM -> decision -> Blynk -> ESP32) without waiting an hour. For results
you report in Ch4, use live_control_loop_1h.py.

Before running:
  1. BLYNK_TOKEN and CURRENT_APPLIANCE set in live_control_loop.py
  2. ESP32 sending V0 often: in ilyasa_esp32_main.ino set
         LIVE_SEND_MS = 5000UL
     (or run within 10 min of ESP32 power-up, when it sends every 10 s anyway)

Run:
    python live_control_fast.py                # 30 s steps, 150 kWh month so far
    python live_control_fast.py --step 20 --kwh 195
Stop with Ctrl+C. Decisions are logged to data/live_fast/fast_test_log.csv.
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

import live_control_loop as loop
import blynk_chart
from control_logic import decide

BASE_DIR = Path(__file__).resolve().parent.parent
loop.INPUT_STEPS = 4
loop.MODELS_DIR = BASE_DIR / "models_1h"
loop.PROCESSED_DIR = BASE_DIR / "data" / "processed_1h"
LOG_DIR = BASE_DIR / "data" / "live_fast"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / "fast_test_log.csv"
READINGS = LOG_DIR / "readings.csv"   # every V0 reading, with date (for the dashboard's Today tab)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--appliance", choices=["fan", "geyser", "lamp"], default=None,
                   help="what is plugged in (default: CURRENT_APPLIANCE in live_control_loop.py)")
    p.add_argument("--step", type=int, default=30, help="seconds between readings (default 30)")
    p.add_argument("--no-scaling", action="store_true", help="turn the wattage mapping off")
    p.add_argument("--tariff", choices=["tou", "a"], default=None,
                   help="tou = TNB Time of Use, a = normal tariff (default: Blynk V19 switch, else ToU)")
    p.add_argument("--kwh", type=float, default=None,
                   help="cumulative kWh this month (default: MANUAL_CUMULATIVE_KWH or 150)")
    a = p.parse_args()

    appliance = a.appliance or loop.CURRENT_APPLIANCE
    if a.no_scaling:
        loop.set_scaling(False)
    if a.tariff:
        loop.TARIFF_MODE = "TOU" if a.tariff == "tou" else "A"
    kwh = a.kwh if a.kwh is not None else (loop.MANUAL_CUMULATIVE_KWH or 150.0)
    if loop.BLYNK_TOKEN.startswith("YOUR_"):
        raise SystemExit("Set BLYNK_TOKEN in live_control_loop.py first.")

    blynk_chart.install()
    loop.sync_scaling_from_blynk()
    loop.sync_tariff_from_blynk()
    print("=" * 70)
    print(loop.mapping_status(appliance))
    print(loop.tariff_status())
    print(f"FAST TEST MODE: {a.step}s steps (model trained on 15-min steps).")
    print("Tests the live chain; not a real 30-minute forecast. Ctrl+C to stop.")
    print(f"appliance={appliance}  month so far={kwh} kWh  first decision in ~{4 * a.step // 60 or 1} min")
    print("=" * 70)

    if not LOG.exists():
        LOG.write_text("time,appliance,readings_w,forecast_w,threshold_w,action,reason\n")

    window = deque(maxlen=4)
    last_action = None
    same_count = 0
    while True:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        now = stamp[11:]
        try:
            w = loop.blynk_get("V0")
        except (requests.RequestException, ValueError) as e:
            print(f"[{now}] could not read V0 from Blynk: {e}")
            time.sleep(a.step)
            continue

        same_count = same_count + 1 if window and w == window[-1] else 0
        if same_count == 6:
            print("   note: V0 hasn't changed for 6 readings. If the load is steady that's fine; "
                  "otherwise set LIVE_SEND_MS = 5000UL in the ESP32 sketch.")
        window.append(w)
        if not READINGS.exists():
            READINGS.write_text("time,appliance,power_w\n")
        with READINGS.open("a") as f:
            f.write(f"{stamp},{appliance},{w}\n")

        loop.sync_scaling_from_blynk()
        loop.sync_tariff_from_blynk()
        loop.push_tou(w, cost_every_s=300)   # cost every 5 min; period/rate only when they change
        if len(window) < 4:
            print(f"[{now}] reading {len(window)}/4: {w:.1f} W  (collecting, no decision yet)")
        else:
            pred = loop.predict_next(appliance, np.array(window))
            tou = loop.current_tou(kwh)
            d = decide(appliance, w, pred, kwh, tou=tou)
            blynk_chart.push(appliance, d.threshold_w)   # Blynk chart lines V21 LSTM, V22 historical, V23 limit
            tlabel = f"ToU {tou.period}" if tou else "Tariff A"
            print(f"[{now}] now {w:.1f} W | forecast {pred:.1f} W | limit {d.threshold_w:.1f} W "
                  f"-> {d.action.value} ({d.alert_tier.value})  [mapping {'on' if loop.SCALING_ENABLED else 'off'}, {tlabel}]")
            with LOG.open("a") as f:
                f.write(f"{stamp},{appliance},{' '.join(f'{x:.1f}' for x in window)},"
                        f"{pred:.1f},{d.threshold_w:.0f},{d.action.value},\"{d.reason}\"\n")
            if d.action != last_action:  # only push changes, so the buzzer doesn't beep every 30 s
                try:
                    loop.blynk_update(loop.CONTROL_PIN, loop.ACTION_TO_BLYNK_INT[d.action])
                    loop.blynk_update(loop.REASON_PIN, d.reason[:64])
                    print(f"   sent to Blynk: V10={loop.ACTION_TO_BLYNK_INT[d.action]} ({d.action.value})")
                    last_action = d.action
                except requests.RequestException as e:
                    print(f"   Blynk write failed: {e}")
        time.sleep(a.step)


if __name__ == "__main__":
    main()
