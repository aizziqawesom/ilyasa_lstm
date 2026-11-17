"""
Week 4 deliverable: the live control loop. This is the piece SETUP_AND_TEST.md
flagged as "not built yet" -- it's what actually connects the trained LSTM
model to the live PZEM reading and the physical relay, on a schedule.

ARCHITECTURE (confirmed against Blynk's real HTTP API docs, not guessed):

    PZEM-004T --(UART)--> ESP32 --(WiFi, Blynk.begin())--> Blynk cloud
                                                                 ^
                                                                 |  HTTP GET (this script polls
                                                                 |  the current reading, and later
                                                                 |  writes the decision back)
                                                                 v
                                            THIS SCRIPT (laptop) -- loads the trained LSTM,
                                            calls control_logic.decide(), pushes the result

    Blynk cloud --(WiFi, BLYNK_WRITE(V_CONTROL) callback)--> ESP32 --> relay / buzzer / LCD

This script NEVER talks to the ESP32 directly. It only talks to Blynk over
plain HTTPS (no persistent connection, no embedded Blynk library needed on
the Python side) -- Blynk is the message broker in both directions. That
answers the "do we send to Blynk then to ESP32" question directly: yes,
always via Blynk, exactly like the existing diagrams show.

Blynk endpoints used (verified against https://docs.blynk.io/en/blynk.cloud/
device-https-api, current as of Sept 2026):
    GET https://blynk.cloud/external/api/get?token={token}&{pin}
        -> current value of a virtual pin (used to poll the live PZEM reading)
    GET https://blynk.cloud/external/api/update?token={token}&{pin}={value}
        -> write a value to a virtual pin (used to push the decision + reason)

DESIGN CHOICE -- local rolling buffer instead of Blynk's history API:
Blynk does have a /external/api/data/get history endpoint that can return
past readings, but how much history is actually retained/queryable depends
on the Blynk plan on the account, which wasn't something to assume. Instead,
this script builds its OWN rolling window locally: every cycle it polls the
CURRENT reading, appends it to a small local buffer file, and reads back
the last INPUT_STEPS buckets from that buffer. This has zero dependency on
Blynk's history retention and survives fine across script restarts (the
buffer is a file, not in-memory state) -- the only cost is a warm-up period
of INPUT_STEPS * POLL_INTERVAL (2 hours) after first starting the script,
during which it polls and logs but doesn't yet have enough data to predict.

DESIGN CHOICE -- 2-step forecast collapsed to one number via max(), not mean():
train_lstm.py's models output OUTPUT_STEPS=2 values (the next 30 minutes in
two 15-min steps). control_logic.decide() wants a single predicted_power_w.
Using max() of the two steps is the conservative choice for a safety-relevant
appliance (geyser) -- it reacts to a spike appearing in EITHER of the next
two windows, not just the first one. Documented here because it's a real
design decision, not an obvious default.

KNOWN LIMITATION -- cumulative_kwh_this_month is a proxy, not the real TNB
bill figure: this script can only see the ONE appliance currently plugged
into the monitored circuit, not the household's actual smart meter. It
accumulates the monitored appliance's own energy use as a stand-in for
"how close is the household to the next TNB tier" -- which is the same
simplification the whole control-logic design has used since Week 3, just
now actually running instead of only simulated. If Ilyasa has his real TNB
cumulative figure (e.g. checks his bill/meter), MANUAL_CUMULATIVE_KWH below
can be set to override the auto-tracked proxy -- state this plainly in the
report rather than implying the script reads the household's real meter.

HARDWARE SCOPE (per the current decision): one PZEM-004T + one relay,
reused sequentially between the water heater and the fan by physically
swapping which appliance is plugged into the socket box. CURRENT_APPLIANCE
below is a manual switch -- change it to match whatever's actually plugged
in before starting a session. This script does not attempt to auto-detect
which appliance is connected.

STILL NEEDED before this runs for real:
    1. BLYNK_TOKEN below -- same device auth token as BLYNK_AUTH_TOKEN in
       firmware/ilyasa_esp32_main/ilyasa_esp32_main.ino.
    POWER_PIN is V5, the 15-minute average power that ilyasa_esp32_main.ino
    publishes. Using the average (not the instantaneous V0 reading) means
    each 15-min bucket in the rolling window is a real average, which
    matters for the water heater's on/off cycling.
"""

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_logic  # noqa: E402
import tou_pricing  # noqa: E402
from control_logic import Action, decide  # noqa: E402

# ---- Fill in / confirm before running for real ----
BLYNK_TOKEN = "xRV6p-qoKH0DTmRQplfMyJu5V3wowTEn"      # same token as BLYNK_AUTH_TOKEN in ilyasa_esp32_main.ino
BLYNK_SERVER = "sgp1.blynk.cloud"          # account region is SGP1 (shown in Blynk console footer)
POWER_PIN = "V5"                            # 15-min average power from ilyasa_esp32_main.ino
CONTROL_PIN = "V10"                         # matches V_CONTROL in relay_control.ino
REASON_PIN = "V11"                          # matches V_STATUS_TEXT in relay_control.ino

# ---- Manual appliance selector -- change to match what's physically plugged in ----
CURRENT_APPLIANCE = "geyser"                # "geyser" or "fan" (laptop has no live hardware)

# ---- Timing, matches preprocess.py / train_lstm.py conventions ----
POLL_INTERVAL_SEC = 15 * 60                 # 15 min, matches RESAMPLE_FREQ in preprocess.py
INPUT_STEPS = 8                             # 2 hours of 15-min buckets, matches train_lstm.py
OUTPUT_STEPS = 2                            # matches train_lstm.py

# ---- Optional manual override for the real household cumulative kWh ----
# Set to a number (e.g. from Ilyasa's actual TNB meter/bill) to use that
# instead of the auto-tracked single-appliance proxy. Leave as None to use
# the proxy.
MANUAL_CUMULATIVE_KWH = None

# ---- Wattage scaling: real appliance vs the dataset appliance it was trained on ----
# The fan model was trained on the iAWE TV channel, which runs at about 74 W.
# If your real fan runs at a different power (e.g. 50 W), the model is being
# fed a power level it never saw. Scaling fixes the SIZE mismatch:
#   1. readings are multiplied by (TRAIN_TYPICAL_W / REAL_TYPICAL_W) before the model
#   2. the forecast is divided by the same factor, back to real watts
#   3. the appliance's control threshold is divided by the same factor, so the
#      decision rules keep the same meaning (e.g. fan 85 W limit -> about 57 W)
# It does NOT fix a different usage PATTERN, only the power level. State this
# in the report ("amplitude-normalised to the training appliance's level").
# Put your appliance's normal running power here (read it from the Serial
# Monitor). Remove an entry to switch scaling off for that appliance.
REAL_TYPICAL_W = {
    "fan": 50.0,        # Ilyasa's real fan (measured)
    "lamp": 3.0,        # 3 W test lamp (measured about 3 W at 243 V)
}
TRAIN_TYPICAL_W = {     # median running power of each training channel (iAWE, see control_logic.py)
    "fan": 74.0,        # TV channel
    "geyser": 1758.0,   # air-con channel
    "computer": 34.0,   # laptop channel
    "lamp": 74.0,       # lamp borrows the fan's model, so it maps to the same TV channel
}

# Which trained model each appliance uses. The lamp has no lamp data in iAWE,
# so it uses the fan model: the TV channel behind it is also a steady load
# that a person switches on and off, the closest behaviour to a lamp.
MODEL_FOR = {"lamp": "fan"}
# geyser is NOT in REAL_TYPICAL_W on purpose: its threshold was already
# recalibrated for the real 1500 W heater (1620 W). Scaling it too would
# double-adjust. If you add it, set its threshold back to 1900 W first.


# Mapping on/off. Switched live from Blynk V15 (1 = on, 0 = off) every cycle,
# or start with --no-scaling. When off, the model sees raw watts and the
# thresholds go back to their unscaled values.
SCALING_ENABLED = True
SCALING_PIN = "V15"
TEMP_PIN, HUMIDITY_PIN = "V6", "V7"   # DHT22 from the ESP32 (logged, not used in decisions yet)


def scale_factor(appliance: str) -> float:
    if not SCALING_ENABLED:
        return 1.0
    if appliance in REAL_TYPICAL_W and REAL_TYPICAL_W[appliance] > 0:
        return TRAIN_TYPICAL_W[appliance] / REAL_TYPICAL_W[appliance]
    return 1.0


_BASE_THRESHOLDS = dict(control_logic.APPLIANCE_THRESHOLDS_W)


def set_scaling(enabled: bool) -> None:
    """Turn the wattage mapping on/off and set the thresholds to match."""
    global SCALING_ENABLED
    changed = SCALING_ENABLED != enabled
    SCALING_ENABLED = enabled
    for a in REAL_TYPICAL_W:
        control_logic.APPLIANCE_THRESHOLDS_W[a] = _BASE_THRESHOLDS[a] / scale_factor(a)
    if changed:
        print(f"   wattage mapping is now {'ON' if enabled else 'OFF'}")


set_scaling(True)


_last_scaling_error = None


def sync_scaling_from_blynk() -> None:
    """Read V15 (1 on / 0 off). Keeps the current setting if it can't be read, and says why (once)."""
    global _last_scaling_error
    if "--no-scaling" in sys.argv:
        return
    if BLYNK_TOKEN.startswith("YOUR_"):
        if _last_scaling_error != "token":
            print("   mapping: BLYNK_TOKEN not set in live_control_loop.py, so the Blynk switch is ignored "
                  f"(mapping stays {'ON' if SCALING_ENABLED else 'OFF'})")
            _last_scaling_error = "token"
        return
    try:
        set_scaling(int(float(blynk_get(SCALING_PIN))) != 0)
        _last_scaling_error = None
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if msg != _last_scaling_error:
            print(f"   mapping: could not read the V15 switch ({msg}); "
                  f"keeping mapping {'ON' if SCALING_ENABLED else 'OFF'}. "
                  "If the switch was never pressed, press it once in Blynk.")
            _last_scaling_error = msg


def mapping_status(appliance: str) -> str:
    """One-line summary for start-up messages."""
    if appliance not in REAL_TYPICAL_W:
        return (f"wattage mapping: not used for '{appliance}' (no entry in REAL_TYPICAL_W, "
                "the model already matches this appliance's power level)")
    return (f"wattage mapping: {'ON' if SCALING_ENABLED else 'OFF'} for {appliance} "
            f"({REAL_TYPICAL_W[appliance]:g} W real -> {TRAIN_TYPICAL_W[appliance]:g} W training, "
            f"x{TRAIN_TYPICAL_W[appliance] / REAL_TYPICAL_W[appliance]:.2f}); switch it with Blynk V15")


# ---- Tariff: "TOU" (TNB Time of Use, peak 2pm-10pm weekdays) or "A" (normal tariff) ----
# Switched live from Blynk V19 (1 = ToU, 0 = Tariff A), or start with --tariff tou / --tariff a.
TARIFF_MODE = "TOU"
TARIFF_PIN = "V19"
TOU_PERIOD_PIN, TOU_RATE_PIN, COST_HOUR_PIN, COST_TODAY_PIN = "V16", "V17", "V18", "V20"
_tou_state = {"pushed": {}, "last_cost_push": 0.0, "day": None, "cost_today": 0.0, "last_t": None}


def current_tou(monthly_kwh: float = 0.0):
    """TouInfo when ToU mode is on, else None (control_logic then uses the Tariff A brackets)."""
    return tou_pricing.tou_info(monthly_kwh=monthly_kwh) if TARIFF_MODE == "TOU" else None


def sync_tariff_from_blynk() -> None:
    global TARIFF_MODE
    if "--tariff" in sys.argv or BLYNK_TOKEN.startswith("YOUR_"):
        return
    try:
        new = "TOU" if int(float(blynk_get(TARIFF_PIN))) == 1 else "A"
    except Exception:  # noqa: BLE001 - switch never pressed / not created: keep current mode
        return
    if new != TARIFF_MODE:
        print(f"   tariff is now {'ToU (peak 2pm-10pm weekdays)' if new == 'TOU' else 'Tariff A (monthly brackets)'}")
        TARIFF_MODE = new


def tariff_status() -> str:
    if TARIFF_MODE != "TOU":
        return "tariff: Tariff A (normal TNB, monthly kWh brackets). Switch with Blynk V19 or --tariff tou"
    i = tou_pricing.tou_info()
    return (f"tariff: TNB ToU, now {i.period} ({i.all_in_sen:.2f} sen/kWh all-in), {i.label}. "
            "Switch with Blynk V19 or --tariff a")


def push_tou(power_w: float, cost_every_s: float = 60.0) -> None:
    """Send ToU period / rate / cost to Blynk. Unchanged values are not re-sent (saves messages)."""
    st = _tou_state
    now = time.time()
    info = tou_pricing.tou_info()
    # running cost today, from the readings this script sees (only while the script is running)
    today = datetime.now(tou_pricing.MYT).date()
    if st["day"] != today:
        st["day"], st["cost_today"] = today, 0.0
    if st["last_t"] is not None:
        hrs = min(now - st["last_t"], 3600.0) / 3600.0
        price = info if TARIFF_MODE == "TOU" else None
        sen = price.all_in_sen if price else tou_pricing.GENERAL_ALL_IN_SEN
        st["cost_today"] += power_w / 1000.0 * hrs * sen / 100.0
    st["last_t"] = now
    if BLYNK_TOKEN.startswith("YOUR_"):
        return
    if TARIFF_MODE == "TOU":
        vals = {TOU_PERIOD_PIN: info.label, TOU_RATE_PIN: info.all_in_sen}
    else:
        vals = {TOU_PERIOD_PIN: "Normal tariff, no peak", TOU_RATE_PIN: tou_pricing.GENERAL_ALL_IN_SEN}
    if now - st["last_cost_push"] >= cost_every_s:
        rate = info.all_in_sen if TARIFF_MODE == "TOU" else tou_pricing.GENERAL_ALL_IN_SEN
        vals[COST_HOUR_PIN] = round(power_w / 1000.0 * rate / 100.0, 4)
        vals[COST_TODAY_PIN] = round(st["cost_today"], 3)
        st["last_cost_push"] = now
    try:
        for pin, v in vals.items():
            if pin in (COST_HOUR_PIN, COST_TODAY_PIN) or st["pushed"].get(pin) != v:
                blynk_update(pin, v)
                st["pushed"][pin] = v
    except requests.RequestException as e:
        print(f"   ToU push to Blynk failed: {e}")


def read_dht() -> tuple[float | None, float | None]:
    """Latest DHT22 values from Blynk V6/V7, or (None, None) if not available."""
    try:
        return float(blynk_get(TEMP_PIN)), float(blynk_get(HUMIDITY_PIN))
    except Exception:  # noqa: BLE001
        return None, None

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LIVE_DIR = BASE_DIR / "data" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

ACTION_TO_BLYNK_INT = {
    Action.OK: 0,
    Action.NOTIFY: 1,
    Action.SHIFT: 2,
    Action.CUTOFF: 3,
}

_model_cache = {}
_scaler_cache = {}


def buffer_path(appliance: str) -> Path:
    return LIVE_DIR / f"{appliance}_buffer.csv"


def state_path(appliance: str) -> Path:
    return LIVE_DIR / f"{appliance}_state.json"


def blynk_get(pin: str) -> float:
    """GET https://blynk.cloud/external/api/get?token=...&{pin} -- current value of a pin."""
    # Exact format from Blynk's docs: ...get?token=XXX&V15 (pin with no "=" after it)
    url = f"https://{BLYNK_SERVER}/external/api/get?token={BLYNK_TOKEN}&{pin}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Blynk said {resp.status_code}: {resp.text.strip()[:120]}")
    return float(resp.text)


def blynk_update(pin: str, value) -> None:
    """GET https://blynk.cloud/external/api/update?token=...&{pin}={value} -- write a pin."""
    url = f"https://{BLYNK_SERVER}/external/api/update"
    resp = requests.get(url, params={"token": BLYNK_TOKEN, pin: value}, timeout=10)
    resp.raise_for_status()


def append_reading(appliance: str, power_w: float, ts: datetime) -> None:
    path = buffer_path(appliance)
    row = pd.DataFrame([{"timestamp": ts.isoformat(), "power_w": power_w}])
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)


def load_recent_window(appliance: str) -> np.ndarray | None:
    """Read the buffer file back, resample to 15-min buckets, return the last
    INPUT_STEPS buckets as a 1D array of Watts, or None if there isn't enough
    history yet (still warming up)."""
    path = buffer_path(appliance)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return None
    df = df.set_index("timestamp").sort_index()
    resampled = df["power_w"].resample("15min").mean().ffill(limit=2)
    resampled = resampled.dropna()
    if len(resampled) < INPUT_STEPS:
        return None
    return resampled.iloc[-INPUT_STEPS:].to_numpy()


def get_model_and_scaler(appliance: str):
    if appliance not in _model_cache:
        from tensorflow.keras.models import load_model

        src = MODEL_FOR.get(appliance, appliance)
        _model_cache[appliance] = load_model(MODELS_DIR / f"lstm_{src}.keras")
        with open(PROCESSED_DIR / src / "scaler.pkl", "rb") as f:
            _scaler_cache[appliance] = pickle.load(f)
    return _model_cache[appliance], _scaler_cache[appliance]


def predict_next(appliance: str, window_w: np.ndarray) -> float:
    """Run the trained model on the last INPUT_STEPS readings, return a single
    predicted_power_w for control_logic.decide() -- max() of the two forecast
    steps (see module docstring for why max, not mean)."""
    model, scaler = get_model_and_scaler(appliance)
    k = scale_factor(appliance)  # 1.0 unless REAL_TYPICAL_W is set for this appliance
    x_scaled = scaler.transform((window_w * k).reshape(-1, 1)).reshape(1, INPUT_STEPS, 1)
    y_scaled = model.predict(x_scaled, verbose=0)
    y_w = scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(OUTPUT_STEPS) / k
    return float(np.max(y_w))


def load_cumulative_kwh(appliance: str) -> float:
    if MANUAL_CUMULATIVE_KWH is not None:
        return MANUAL_CUMULATIVE_KWH
    path = state_path(appliance)
    now_month = datetime.now().strftime("%Y-%m")
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("month") == now_month:
            return state.get("cumulative_kwh", 0.0)
    return 0.0


def save_cumulative_kwh(appliance: str, cumulative_kwh: float) -> None:
    now_month = datetime.now().strftime("%Y-%m")
    state_path(appliance).write_text(json.dumps({"month": now_month, "cumulative_kwh": cumulative_kwh}))


def decisions_path() -> Path:
    return LIVE_DIR / "decisions.csv"


def log_decision(ts, appliance, window_w, current_w, forecast_w, cum_kwh, decision,
                 temp_c=None, humidity=None) -> None:
    """Every decision goes to LIVE_DIR/decisions.csv (read by dashboard.py, and Ch4 evidence)."""
    row = pd.DataFrame([{
        "time": ts.isoformat(),
        "appliance": appliance,
        "window_w": " ".join(f"{x:.1f}" for x in window_w),
        "current_w": round(current_w, 2),
        "forecast_w": round(forecast_w, 2),
        "threshold_w": round(decision.threshold_w, 2),
        "cumulative_kwh": round(cum_kwh, 3),
        "action": decision.action.value,
        "tier": decision.alert_tier.value,
        "scale_factor": round(scale_factor(appliance), 3),
        "scaling": "on" if SCALING_ENABLED else "off",
        "temp_c": temp_c,
        "humidity_pct": humidity,
        "tariff": TARIFF_MODE,
        "tou_period": (tou_pricing.tou_info().period if TARIFF_MODE == "TOU" else ""),
        "reason": decision.reason,
    }])
    path = decisions_path()
    if path.exists():
        with open(path) as f:
            old_header = f.readline().strip().split(",")
        if old_header != list(row.columns):   # older log format: keep it, start a fresh file
            path.rename(path.with_name(f"decisions_old_{datetime.now():%Y%m%d_%H%M%S}.csv"))
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_cycle() -> None:
    appliance = CURRENT_APPLIANCE
    now = datetime.now(timezone.utc)

    try:
        current_power_w = blynk_get(POWER_PIN)
    except requests.RequestException as e:
        print(f"[{now.isoformat()}] Blynk read failed: {e} -- skipping this cycle")
        return

    append_reading(appliance, current_power_w, now)
    window_w = load_recent_window(appliance)

    if window_w is None:
        print(f"[{now.isoformat()}] {appliance}: warming up, not enough history yet "
              f"(need {INPUT_STEPS} x 15-min buckets = {INPUT_STEPS * 15} min) -- reading was {current_power_w:.0f}W")
        return

    sync_scaling_from_blynk()
    sync_tariff_from_blynk()
    push_tou(current_power_w, cost_every_s=0)
    predicted_power_w = predict_next(appliance, window_w)

    # Update the cumulative-kWh proxy (see module docstring caveat about this
    # being a single-appliance proxy, not the household's real TNB figure)
    cumulative_kwh = load_cumulative_kwh(appliance)
    if MANUAL_CUMULATIVE_KWH is None:
        cumulative_kwh += (current_power_w / 1000.0) * (POLL_INTERVAL_SEC / 3600.0)
        save_cumulative_kwh(appliance, cumulative_kwh)

    decision = decide(appliance, current_power_w, predicted_power_w, cumulative_kwh,
                      tou=current_tou(cumulative_kwh))

    print(f"[{now.isoformat()}] {appliance}: cur={current_power_w:.0f}W pred={predicted_power_w:.0f}W "
          f"cum={cumulative_kwh:.1f}kWh -> {decision.action.value} ({decision.alert_tier.value}) "
          f"| {decision.reason}")

    temp_c, humidity = read_dht()
    log_decision(now, appliance, window_w, current_power_w, predicted_power_w, cumulative_kwh, decision,
                 temp_c, humidity)

    try:
        blynk_update(CONTROL_PIN, ACTION_TO_BLYNK_INT[decision.action])
        blynk_update(REASON_PIN, decision.reason[:64])
    except requests.RequestException as e:
        print(f"[{now.isoformat()}] Blynk write failed: {e} -- decision was computed but NOT sent to ESP32")


def main() -> None:
    global CURRENT_APPLIANCE, TARIFF_MODE
    run_once = "--once" in sys.argv
    if "--tariff" in sys.argv:
        TARIFF_MODE = "TOU" if sys.argv[sys.argv.index("--tariff") + 1].lower() == "tou" else "A"
    if "--no-scaling" in sys.argv:
        set_scaling(False)
    if "--appliance" in sys.argv:
        CURRENT_APPLIANCE = sys.argv[sys.argv.index("--appliance") + 1]
    if CURRENT_APPLIANCE not in ("geyser", "fan", "lamp", "computer"):
        raise SystemExit(f"Unknown appliance '{CURRENT_APPLIANCE}'. Use geyser, fan or lamp.")
    sync_scaling_from_blynk()
    sync_tariff_from_blynk()
    print(mapping_status(CURRENT_APPLIANCE))
    print(tariff_status())
    print(f"Live control loop starting. Appliance={CURRENT_APPLIANCE}, "
          f"poll every {POLL_INTERVAL_SEC // 60} min. Ctrl+C to stop.")
    while True:
        run_cycle()
        if run_once:
            break
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
