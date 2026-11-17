"""
Data for the dashboard's "Today" tab: one day (00:00-24:00 Malaysia time) in
15-minute buckets, for one appliance, with three power series and their cost:

  actual      today's real readings (PZEM -> Blynk -> the Python loops' log files)
  historical  the average day in the iAWE training data for this appliance's model,
              divided by the wattage scale factor so it is in the real appliance's watts
  lstm        the LSTM's expected power for every bucket of the day:
                - up to now: one-step-ahead forecast from the previous window
                  (actual readings where we have them, the historical average where we don't)
                - after now: rolled forward, feeding its own forecasts back in,
                  starting from the latest real readings

Cost per series = power x price at that time of day (TNB ToU or the normal tariff).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import live_control_loop as loop
import tou_pricing as T

BUCKETS = 96   # 15-min buckets in a day


def today_myt() -> datetime:
    return datetime.now(T.MYT).replace(tzinfo=None)


def day_index(day: datetime) -> pd.DatetimeIndex:
    start = pd.Timestamp(day.date())
    return pd.date_range(start, periods=BUCKETS, freq="15min")


def _to_myt_naive(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return t.dt.tz_convert("Asia/Kuala_Lumpur").dt.tz_localize(None)


def actual_readings(base: Path, appliance: str) -> pd.DataFrame:
    """Every live reading we have for this appliance: time (Malaysia, naive), power_w, source."""
    frames = []
    for live in ("live", "live_1h"):
        p = base / "data" / live / f"{appliance}_buffer.csv"
        if p.exists() and p.stat().st_size:
            b = pd.read_csv(p)
            frames.append(pd.DataFrame({"time": _to_myt_naive(b["timestamp"]), "power_w": b["power_w"],
                                        "source": f"{live} loop"}))
    p = base / "data" / "live_fast" / "readings.csv"          # every V0 reading the fast loop saw
    if p.exists() and p.stat().st_size:
        r = pd.read_csv(p)
        r = r[r["appliance"] == appliance]
        frames.append(pd.DataFrame({"time": pd.to_datetime(r["time"], errors="coerce"),
                                    "power_w": r["power_w"], "source": "fast loop"}))
    if not frames:
        return pd.DataFrame(columns=["time", "power_w", "source"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["time", "power_w"])
    return out.sort_values("time")


def historical_profile(processed_dir: Path, appliance: str) -> np.ndarray | None:
    """Average iAWE power for each 15-min slot of the day, in the REAL appliance's watts."""
    src = loop.MODEL_FOR.get(appliance, appliance)
    p = processed_dir / src / "resampled_15min.csv"
    if not p.exists():
        return None
    ds = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp")["power_w"]
    slot = ds.index.hour * 4 + ds.index.minute // 15
    prof = ds.groupby(slot).mean().reindex(range(BUCKETS)).interpolate(limit_direction="both")
    return prof.to_numpy() / loop.scale_factor(appliance)


def lstm_day(model, scaler, steps: int, appliance: str, hist: np.ndarray,
             actual_b: np.ndarray, now_bucket: int) -> np.ndarray:
    """LSTM expected power for all 96 buckets (see module docstring). NaN for the first `steps`
    buckets of the day, which have no window before them (they're filled from yesterday's
    historical average so the curve still starts at 00:00)."""
    k = loop.scale_factor(appliance)
    # the series the model looks back on: actual where known, else historical average
    known = np.where(np.isnan(actual_b), hist, actual_b)
    # prepend yesterday's evening (historical) so 00:00 has a full window
    series = np.concatenate([hist[-steps:], known])

    def run(windows: np.ndarray) -> np.ndarray:
        x = scaler.transform((windows * k).reshape(-1, 1)).reshape(len(windows), steps, 1)
        y = model(x, training=False).numpy() if hasattr(model, "__call__") else model.predict(x, verbose=0)
        return scaler.inverse_transform(y.reshape(-1, 1)).reshape(len(windows), -1) / k

    out = np.full(BUCKETS, np.nan)
    # past (and the current bucket): one-step-ahead, all in one batch
    last_past = min(now_bucket, BUCKETS - 1)
    idx = np.arange(0, last_past + 1)
    wins = np.stack([series[i:i + steps] for i in idx])
    out[idx] = run(wins)[:, 0]
    # future: roll forward from now, 2 buckets per model call
    roll = list(series[: last_past + steps + 1])   # series up to and including the current bucket
    roll[-1] = known[last_past] if not np.isnan(actual_b[last_past]) else out[last_past]
    b = last_past + 1
    while b < BUCKETS:
        y = run(np.array([roll[-steps:]]))[0]
        for v in y:
            if b < BUCKETS:
                out[b] = max(v, 0.0)
                roll.append(max(v, 0.0))
                b += 1
    return np.clip(out, 0, None)


def price_sen(index: pd.DatetimeIndex, tariff: str) -> np.ndarray:
    """All-in price (sen/kWh) for each bucket, ToU or normal tariff."""
    if tariff != "TOU":
        return np.full(len(index), T.GENERAL_ALL_IN_SEN)
    return np.array([T.tou_info(t.to_pydatetime().replace(tzinfo=T.MYT)).all_in_sen for t in index])


def build_day(model, scaler, steps: int, processed_dir: Path, base: Path, appliance: str,
              tariff: str, now: datetime | None = None):
    """Returns (frame, readings): frame has one row per 15-min bucket with
    actual_w / historical_w / lstm_w, price_sen and the matching cost_*_rm_h columns."""
    now = now or today_myt()
    idx = day_index(now)
    hist = historical_profile(processed_dir, appliance)
    if hist is None:
        return None, None
    rd = actual_readings(base, appliance)
    rd_today = rd[(rd["time"] >= idx[0]) & (rd["time"] < idx[0] + timedelta(days=1))]
    actual_b = (rd_today.set_index("time")["power_w"].resample("15min").mean()
                .reindex(idx).to_numpy() if not rd_today.empty else np.full(BUCKETS, np.nan))
    now_bucket = int((now - idx[0].to_pydatetime()).total_seconds() // 900)
    lstm = lstm_day(model, scaler, steps, appliance, hist, actual_b, now_bucket)
    price = price_sen(idx, tariff)
    f = pd.DataFrame({"time": idx, "actual_w": actual_b, "historical_w": hist, "lstm_w": lstm,
                      "price_sen": price})
    for c in ("actual", "historical", "lstm"):
        f[f"cost_{c}_rm_h"] = f[f"{c}_w"] / 1000.0 * f["price_sen"] / 100.0
    f["is_future"] = np.arange(BUCKETS) > now_bucket
    return f, rd_today


def day_totals(f: pd.DataFrame) -> dict:
    """kWh and RM for the whole day (each bucket is 0.25 h)."""
    out = {}
    for c in ("actual", "historical", "lstm"):
        w = f[f"{c}_w"]
        out[c] = dict(kwh=float(np.nansum(w) * 0.25 / 1000),
                      rm=float(np.nansum(f[f"cost_{c}_rm_h"]) * 0.25))
    # "expected total today" = actual so far + LSTM for the rest of the day
    past = ~f["is_future"]
    so_far_rm = float(np.nansum(f.loc[past, "cost_actual_rm_h"]) * 0.25)
    rest_rm = float(np.nansum(f.loc[~past, "cost_lstm_rm_h"]) * 0.25)
    out["expected_today_rm"] = so_far_rm + rest_rm
    out["so_far_rm"] = so_far_rm
    return out
