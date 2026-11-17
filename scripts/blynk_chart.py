"""
Extra lines for the Blynk "Power history" chart, next to V0 (actual power):

  V21 "LSTM expected"  = what the LSTM said ONE STEP AGO the power would be NOW.
                         Sent one step late on purpose: Blynk plots a value at the time it
                         arrives, so a forecast sent when it is made would sit 15-30 min to the
                         left of the reading it predicts. Delayed, it lines up with V0.
  V22 "Historical avg" = the iAWE training-data average for this 15-min slot of the day,
                         in the real appliance's watts (same as the grey line on the
                         dashboard's Today tab).
  V23 "Limit"          = the threshold the decision used at this moment, in real watts (it drops
                         near a TNB bracket or in ToU peak). When orange goes above it, the rules act.

Used by live_control_fast.py and live_control_loop_1h.py. Kept in its own file so
live_control_loop.py (where BLYNK_TOKEN lives) does not need editing.
If V21/V22 don't exist in Blynk yet, it prints one note and the control loop carries on.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import requests

import live_control_loop as loop
import tou_pricing

LSTM_PIN, HIST_PIN, LIMIT_PIN = "V21", "V22", "V23"
_state = {"next_step_w": None, "pending_w": None, "hist": {}, "warned": False}


def install() -> None:
    """Wrap loop.predict_next so every forecast also remembers its first step (next 15 min)
    and, in the 15-min loop, push the chart once per cycle (just before the decision is logged)."""
    if getattr(loop, "_chart_installed", False):
        return
    orig_predict, orig_log = loop.predict_next, loop.log_decision

    def predict_next(appliance, window_w):
        model, scaler = loop.get_model_and_scaler(appliance)
        k = loop.scale_factor(appliance)
        x = scaler.transform((np.asarray(window_w, float) * k).reshape(-1, 1)).reshape(1, loop.INPUT_STEPS, 1)
        y = scaler.inverse_transform(model.predict(x, verbose=0).reshape(-1, 1)).ravel() / k
        _state["next_step_w"] = float(y[0])
        return float(np.max(y))   # same value the original returns (max of the 2 steps)

    def log_decision(ts, appliance, *args, **kwargs):
        decision = args[4] if len(args) > 4 else kwargs.get("decision")
        push(appliance, getattr(decision, "threshold_w", None))
        return orig_log(ts, appliance, *args, **kwargs)

    loop.predict_next, loop.log_decision = predict_next, log_decision
    loop._chart_installed = True
    _state["orig_predict"] = orig_predict


def historical_now_w(appliance: str) -> float | None:
    if appliance not in _state["hist"]:
        src = loop.MODEL_FOR.get(appliance, appliance)
        p = loop.PROCESSED_DIR / src / "resampled_15min.csv"
        prof = None
        if p.exists():
            ds = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp")["power_w"]
            slot = ds.index.hour * 4 + ds.index.minute // 15
            prof = ds.groupby(slot).mean().reindex(range(96)).interpolate(limit_direction="both").to_numpy()
        _state["hist"][appliance] = prof
    prof = _state["hist"][appliance]
    if prof is None:
        return None
    now = datetime.now(tou_pricing.MYT)
    return float(prof[now.hour * 4 + now.minute // 15]) / loop.scale_factor(appliance)


def push(appliance: str, threshold_w: float | None = None) -> None:
    """Call once per cycle, after the decision. Sends V21 (lined-up LSTM), V22 (historical), V23 (limit)."""
    if loop.BLYNK_TOKEN.startswith("YOUR_"):
        return
    vals = {}
    if _state["pending_w"] is not None:
        vals[LSTM_PIN] = round(max(_state["pending_w"], 0.0), 2)
    _state["pending_w"] = _state["next_step_w"]
    hist = historical_now_w(appliance)
    if hist is not None:
        vals[HIST_PIN] = round(hist, 2)
    if threshold_w is not None:
        vals[LIMIT_PIN] = round(float(threshold_w), 2)
    for pin, v in vals.items():
        try:
            loop.blynk_update(pin, v)
        except requests.RequestException as e:
            if not _state["warned"]:
                print(f"   chart push to Blynk {pin} failed ({e}). Create datastreams V21, V22 and V23 in Blynk "
                      "(see RUN_GUIDE_SIMPLE.md); the control loop keeps working without them.")
                _state["warned"] = True
