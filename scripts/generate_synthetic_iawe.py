"""
Generates synthetic per-second appliance power data that mimics the iAWE
(Indian Appliance-level Wireless Electricity) dataset's known structure and
characteristics, for pipeline development/testing BEFORE the real iAWE CSVs
are obtained.

Real iAWE structure (per Batra et al. 2013):
- 73 days of data (we simulate 14 days here for speed; swap DAYS below)
- 1-second resolution power readings (Watts) per appliance channel
- Channels relevant to this project: Geyser (water heater), Fan, Computer (laptop proxy)
- Known data quality issue: intermittent gaps (sensor dropouts), especially on Geyser channel

Once real iAWE data is downloaded (requires request via https://iawe.github.io/),
replace the three output CSVs in data/raw/ with the real per-appliance channel
files and SKIP this script entirely. Column format (timestamp, power_w) must match.
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
DAYS = 14                      # real iAWE = 73 days; smaller here for fast dev/testing
SECONDS = DAYS * 24 * 3600
START = pd.Timestamp("2025-01-01 00:00:00")
timestamps = pd.date_range(START, periods=SECONDS, freq="s")


def hour_of(ts_index):
    return (ts_index.astype("int64") // 10**9 // 3600) % 24


def make_fan(ts):
    """Kipas: long smooth continuous draw, evening/night biased, 30-70W."""
    h = ts.hour + ts.minute / 60
    # usage probability peaks in evening (19:00-23:00) and is low midday
    usage_prob = 0.15 + 0.55 * np.exp(-0.5 * ((h - 21) / 3) ** 2)
    on_mask = RNG.random(len(ts)) < usage_prob
    base = RNG.normal(50, 8, len(ts))
    power = np.clip(base, 30, 70) * on_mask
    return power


def make_geyser(ts):
    """Water heater: sparse bursty pulses, 2000-3000W, ~2-3 short pulses/day."""
    power = np.zeros(len(ts))
    n_days = len(ts) // 86400 + 1
    for d in range(n_days):
        n_pulses = RNG.integers(1, 4)
        for _ in range(n_pulses):
            start_sec = d * 86400 + RNG.integers(0, 86400 - 900)
            if start_sec >= len(ts):
                continue
            dur = RNG.integers(180, 900)  # 3-15 min burst
            end_sec = min(start_sec + dur, len(ts))
            if end_sec <= start_sec:
                continue
            level = RNG.uniform(2000, 3000)
            power[start_sec:end_sec] = level + RNG.normal(0, 50, end_sec - start_sec)
    # inject known-issue missing gaps (sensor dropout) on this channel
    n_gaps = RNG.integers(2, 6)
    for _ in range(n_gaps):
        gap_start = RNG.integers(0, max(len(ts) - 3600, 1))
        gap_len = RNG.integers(600, 3600)
        gap_end = min(gap_start + gap_len, len(ts))
        power[gap_start:gap_end] = np.nan
    return power


def make_computer(ts):
    """Laptop proxy: discontinuous, activity-dependent, 15-60W multi-modal."""
    h = ts.hour + ts.minute / 60
    usage_prob = 0.1 + 0.6 * np.exp(-0.5 * ((h - 14) / 5) ** 2)
    on_mask = RNG.random(len(ts)) < usage_prob
    # bimodal: idle (~15-20W) vs active load (~40-60W)
    mode = RNG.random(len(ts)) < 0.4
    idle = RNG.normal(17, 2, len(ts))
    active = RNG.normal(50, 8, len(ts))
    power = np.where(mode, active, idle)
    power = np.clip(power, 10, 65) * on_mask
    return power


def main():
    print(f"Generating {DAYS} days synthetic iAWE-style data ({SECONDS:,} samples/channel)...")
    channels = {
        "fan": make_fan(timestamps),
        "geyser": make_geyser(timestamps),
        "computer": make_computer(timestamps),
    }
    for name, power in channels.items():
        df = pd.DataFrame({"timestamp": timestamps, "power_w": power})
        path = OUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        n_nan = df["power_w"].isna().sum()
        print(f"  {name}: saved {path.name} | mean={np.nanmean(power):.1f}W "
              f"| max={np.nanmax(power):.1f}W | gaps={n_nan} samples")

    print("Done. Replace these files with real iAWE channel exports when available.")


if __name__ == "__main__":
    main()
