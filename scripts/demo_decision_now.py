"""
INSTANT decision demo - no waiting at all.

Runs the real trained 1-hour LSTM (models_1h/) and the real
control_logic.decide() on a 1-hour input window (4 readings, one per 15 min)
that you choose: either a preset scenario or your own numbers. Prints the
forecast and the decision in a few seconds. Add --push to also send the
decision to Blynk, so the ESP32's LCD and buzzer (and relay, once fitted)
react exactly as they would in the live loop.

Be clear about what this is: the input readings are NOT live sensor data,
they are a scenario. It proves the model + decision + Blynk + ESP32 chain
works on demand (useful for testing and at viva). For decisions from real
live readings use live_control_loop_1h.py.

Examples:
    python demo_decision_now.py --list
    python demo_decision_now.py --scenario fan_normal
    python demo_decision_now.py --scenario heater_near_tariff --push
    python demo_decision_now.py --appliance fan --readings 50 49 50 51 --kwh 150

--push uses BLYNK_TOKEN from live_control_loop.py.
"""

import argparse
from pathlib import Path

from datetime import datetime

import numpy as np

import live_control_loop as loop
import tou_pricing
from control_logic import decide

BASE_DIR = Path(__file__).resolve().parent.parent
loop.INPUT_STEPS = 4
loop.MODELS_DIR = BASE_DIR / "models_1h"
loop.PROCESSED_DIR = BASE_DIR / "data" / "processed_1h"

# appliance, last 4 readings (W, oldest first, 15 min apart), cumulative kWh this month, description
SCENARIOS = {
    "fan_normal":         ("fan",    [50, 49, 50, 51],        150, "Fan (50 W) running normally, mid-month"),
    "fan_near_tariff":    ("fan",    [50, 49, 50, 51],        198, "Same fan, 2 kWh from the 200 kWh TNB bracket"),
    "lamp_normal":        ("lamp",   [3.0, 3.1, 3.0, 3.0],    150, "3 W lamp on, mid-month"),
    "lamp_near_tariff":   ("lamp",   [3.0, 3.1, 3.0, 3.0],    195, "Same lamp, 5 kWh from the 200 kWh TNB bracket (relay cut demo)"),
    "heater_off":         ("geyser", [0, 0, 0, 0],            150, "Water heater off"),
    "heater_normal":      ("geyser", [1500, 1490, 1510, 1500], 150, "Water heater on, mid-month"),
    "heater_near_tariff": ("geyser", [1500, 1490, 1510, 1500], 195, "Same heater, 5 kWh from the 200 kWh TNB bracket"),
}
# ToU scenarios: same readings, only the time of day changes (needs --tou or uses its own)
TOU_SCENARIOS = {
    "fan_offpeak":    ("fan",    [50, 49, 50, 51],         "2026-10-07 11:00", "Fan at 11 am weekday (ToU off-peak)"),
    "fan_peak":       ("fan",    [50, 49, 50, 51],         "2026-10-07 15:00", "Same fan at 3 pm weekday (ToU peak) -> SHIFT"),
    "lamp_offpeak":   ("lamp",   [3.0, 3.1, 3.0, 3.0],     "2026-10-07 11:00", "Lamp at 11 am weekday (ToU off-peak)"),
    "lamp_peak":      ("lamp",   [3.0, 3.1, 3.0, 3.0],     "2026-10-07 15:00", "Same lamp at 3 pm weekday (ToU peak) -> CUTOFF"),
    "heater_offpeak": ("geyser", [1500, 1490, 1510, 1500], "2026-10-07 23:00", "Water heater at 11 pm (ToU off-peak)"),
    "heater_peak":    ("geyser", [1500, 1490, 1510, 1500], "2026-10-07 18:00", "Same heater at 6 pm weekday (ToU peak) -> CUTOFF"),
}


def run(appliance, readings, kwh, push, label="", tou=None):
    window = np.array(readings, dtype=float)
    pred = loop.predict_next(appliance, window)
    d = decide(appliance, float(window[-1]), pred, float(kwh), tou=tou)
    print(f"\n{label or 'custom input'}")
    print(f"  appliance      : {appliance}")
    print(f"  last 1 hour    : {', '.join(f'{r:g}W' for r in readings)}")
    print(f"  month so far   : {kwh} kWh")
    if tou is not None:
        print(f"  tariff         : ToU {tou.period}, {tou.all_in_sen:.2f} sen/kWh all-in, "
              f"running cost RM{tou_pricing.cost_rm(float(window[-1]), 1, tou):.3f}/hour")
    else:
        print("  tariff         : Tariff A (monthly brackets)")
    print(f"  LSTM forecast  : {pred:.1f} W (next 30 min, higher of the 2 steps)")
    print(f"  threshold      : {d.threshold_w:.1f} W (tier-adjusted)")
    print(f"  DECISION       : {d.action.value}  ({d.alert_tier.value})")
    print(f"  reason         : {d.reason}")
    if push:
        if loop.BLYNK_TOKEN.startswith("YOUR_"):
            raise SystemExit("Set BLYNK_TOKEN in live_control_loop.py first.")
        loop.blynk_update(loop.CONTROL_PIN, loop.ACTION_TO_BLYNK_INT[d.action])
        loop.blynk_update(loop.REASON_PIN, d.reason[:64])
        print(f"  sent to Blynk  : V10={loop.ACTION_TO_BLYNK_INT[d.action]}  (watch the LCD / buzzer)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="show preset scenarios")
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--all", action="store_true", help="run every preset scenario")
    p.add_argument("--appliance", choices=["fan", "geyser", "lamp", "computer"])
    p.add_argument("--readings", type=float, nargs=4, metavar="W", help="4 readings, oldest first")
    p.add_argument("--kwh", type=float, default=150, help="cumulative kWh this month (default 150)")
    p.add_argument("--push", action="store_true", help="also send the decision to Blynk")
    p.add_argument("--no-scaling", action="store_true", help="turn the wattage mapping off")
    p.add_argument("--tou", choices=["now", "peak", "offpeak"],
                   help="use TNB ToU pricing: now = real clock, peak = 3 pm weekday, offpeak = 11 am weekday")
    p.add_argument("--tou-scenario", choices=TOU_SCENARIOS, help="preset ToU scenario")
    a = p.parse_args()
    if a.no_scaling:
        loop.set_scaling(False)

    tou = None
    if a.tou:
        when = {"now": None, "peak": "2026-10-07 15:00", "offpeak": "2026-10-07 11:00"}[a.tou]
        tou = tou_pricing.tou_info(None if when is None else
                                   datetime.fromisoformat(when).replace(tzinfo=tou_pricing.MYT))
    if a.list:
        for k, (ap, r, kwh, desc) in SCENARIOS.items():
            print(f"{k:20s} {desc}  [{ap}, {r}, {kwh} kWh]")
        print("ToU scenarios (use --tou-scenario NAME):")
        for k, (ap, r, when, desc) in TOU_SCENARIOS.items():
            print(f"{k:20s} {desc}  [{ap}, {r}, {when}]")
        return
    if a.tou_scenario:
        ap, r, when, desc = TOU_SCENARIOS[a.tou_scenario]
        t = tou_pricing.tou_info(datetime.fromisoformat(when).replace(tzinfo=tou_pricing.MYT))
        run(ap, r, a.kwh, a.push, f"{a.tou_scenario}: {desc}", tou=t)
        return
    if a.all:
        for k, (ap, r, kwh, desc) in SCENARIOS.items():
            run(ap, r, kwh, False, f"{k}: {desc}", tou=tou)
        for k, (ap, r, when, desc) in TOU_SCENARIOS.items():
            t = tou_pricing.tou_info(datetime.fromisoformat(when).replace(tzinfo=tou_pricing.MYT))
            run(ap, r, 150, False, f"{k}: {desc}", tou=t)
        return
    if a.scenario:
        ap, r, kwh, desc = SCENARIOS[a.scenario]
        run(ap, r, kwh, a.push, f"{a.scenario}: {desc}", tou=tou)
        return
    if a.appliance and a.readings:
        run(a.appliance, a.readings, a.kwh, a.push, tou=tou)
        return
    p.error("use --list, --all, --scenario NAME, --tou-scenario NAME, or --appliance X --readings W W W W")


if __name__ == "__main__":
    main()
