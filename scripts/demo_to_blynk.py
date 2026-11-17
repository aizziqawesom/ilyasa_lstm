"""
Send the no-hardware DEMO to Blynk, so the Blynk dashboard (gauges, Power history
chart with V21/V22 lines, decision, reason, cost) shows the demo too.

What it does: makes up a few hours of readings for one appliance (same idea as
simulate_live.py), runs the REAL 1-hour LSTM and the REAL decision rules on them,
and pushes everything to Blynk, one 15-min step every few seconds (time is
compressed: 24 steps = 6 simulated hours, sent in about 2 minutes).

    python demo_to_blynk.py                       # water heater, Tariff A, 150 -> 197 kWh
    python demo_to_blynk.py --appliance fan --tariff tou
    python demo_to_blynk.py --every 3 --steps 32

BEFORE RUNNING: switch the ESP32 OFF (unplug its USB / power).
  - If the ESP32 is online it overwrites V0 every few seconds with the real reading, and
  - it ACTS on the demo decisions (buzzer, relay CUTOFF) -- not what you want in a demo.
  The ESP32 does not replay old decisions when it comes back on, and this script
  sends OK (V10 = 0) at the end anyway.

Honest note for the panel: these are scenario readings, not sensor data. Blynk
plots each value at the time it arrives, so the chart's x-axis shows the ~2 minutes
the demo took, not 6 hours.

Token: uses BLYNK_TOKEN from live_control_loop.py (or the BLYNK_TOKEN environment
variable if set). It is never printed.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import requests

import live_control_loop as loop
import tou_pricing
from control_logic import decide

BASE = Path(__file__).resolve().parent.parent
loop.INPUT_STEPS = 4
loop.MODELS_DIR = BASE / "models_1h"
loop.PROCESSED_DIR = BASE / "data" / "processed_1h"

import blynk_chart  # noqa: E402  (after the 1h paths are set)

# appliance: (typical W, kWh at start, kWh at end) -- same as simulate_live.py
SIM = {"fan": (50, 150, 199), "geyser": (1500, 150, 197), "lamp": (3, 150, 196)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--appliance", choices=list(SIM), default="geyser")
    p.add_argument("--steps", type=int, default=24, help="15-min steps to simulate (default 24 = 6 hours)")
    p.add_argument("--every", type=float, default=5.0, help="real seconds between steps (default 5)")
    p.add_argument("--tariff", choices=["a", "tou"], default="a",
                   help="a = monthly brackets (shows NOTIFY -> CUTOFF as kWh rises), tou = peak/off-peak now")
    p.add_argument("--no-control", action="store_true", help="don't send decisions (V10/V11), only data")
    p.add_argument("--yes", action="store_true", help="skip the 'is the ESP32 off?' question")
    a = p.parse_args()

    if os.environ.get("BLYNK_TOKEN"):
        loop.BLYNK_TOKEN = os.environ["BLYNK_TOKEN"]
    if loop.BLYNK_TOKEN.startswith("YOUR_"):
        raise SystemExit("Set BLYNK_TOKEN in live_control_loop.py first (or the BLYNK_TOKEN environment variable).")
    if not a.yes:
        ans = input("Is the ESP32 switched OFF? (otherwise it overwrites V0 and acts on the decisions) [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            raise SystemExit("Stopped. Unplug the ESP32, then run again.")

    loop.TARIFF_MODE = "TOU" if a.tariff == "tou" else "A"
    blynk_chart.install()
    typ, k0, k1 = SIM[a.appliance]
    rng = np.random.default_rng()
    w = typ * (1 + rng.normal(0, 0.02, a.steps))
    rate = (tou_pricing.tou_info().all_in_sen if loop.TARIFF_MODE == "TOU" else tou_pricing.GENERAL_ALL_IN_SEN)
    cost_today, last_action = 0.0, None

    def send(pin, v):
        try:
            loop.blynk_update(pin, v)
        except requests.RequestException as e:
            print(f"   Blynk {pin} failed: {e}")

    print(f"DEMO -> Blynk: {a.appliance}, {a.steps} steps x 15 simulated min, one every {a.every:g} s, "
          f"tariff {'ToU' if loop.TARIFF_MODE == 'TOU' else 'A (brackets)'}. Ctrl+C to stop.")
    if loop.TARIFF_MODE == "TOU":
        send("V16", tou_pricing.tou_info().label)
    else:
        send("V16", "Normal tariff, no peak")
    send("V17", round(rate, 2))
    send("V11", "DEMO: scenario data, not sensor readings")
    try:
        for i in range(a.steps):
            now_w = round(float(w[i]), 2)
            send("V0", now_w)
            send("V5", now_w)
            cost_today += now_w / 1000 * 0.25 * rate / 100          # 15 simulated minutes
            send("V18", round(now_w / 1000 * rate / 100, 4))
            send("V20", round(cost_today, 3))
            if i < 3:
                print(f"step {i + 1:2d}: {now_w:8.1f} W   (collecting 1 hour of readings)")
            else:
                window = w[i - 3:i + 1]
                kwh = k0 + (k1 - k0) * i / (a.steps - 1)
                pred = loop.predict_next(a.appliance, window)
                tou = loop.current_tou(kwh)
                d = decide(a.appliance, now_w, pred, kwh, tou=tou)
                blynk_chart.push(a.appliance, d.threshold_w)           # V21 + V22 + V23
                print(f"step {i + 1:2d}: {now_w:8.1f} W | forecast {pred:8.1f} W | limit {d.threshold_w:8.1f} W "
                      f"| {kwh:5.1f} kWh -> {d.action.value}")
                if not a.no_control and d.action != last_action:
                    send(loop.CONTROL_PIN, loop.ACTION_TO_BLYNK_INT[d.action])
                    send(loop.REASON_PIN, ("DEMO: " + d.reason)[:64])
                    last_action = d.action
            time.sleep(a.every)
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        if not a.no_control:
            send(loop.CONTROL_PIN, 0)
        send(loop.REASON_PIN, "Demo ended")
        print("Done. Sent OK (V10 = 0) and 'Demo ended'. Switch the ESP32 back on when you want live data.")


if __name__ == "__main__":
    main()
