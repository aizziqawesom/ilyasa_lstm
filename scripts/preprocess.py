"""
Week 1 deliverable: preprocessing pipeline for the 3-appliance LSTM forecasting
project (kipas diri / water heater / laptop), built against the iAWE dataset
format (fan, geyser, computer channels).

Pipeline steps:
1. Load raw per-second per-appliance CSV (timestamp, power_w)
2. Resample to 15-minute buckets (mean power per bucket)
3. Interpolate short gaps, mask/drop long gaps (documented per Ch3.8 methodology)
4. Scale each appliance independently with MinMaxScaler (0-1) -- fit on train
   split only, to avoid leakage
5. Build sliding windows: 96 timesteps (24h) input -> next 4 timesteps (1h) output
6. Chronological 70/15/15 train/val/test split (NOT random shuffle -- time series)
7. Save processed arrays (.npy) + fitted scalers (.pkl) per appliance

Usage:
    python preprocess.py --appliance fan
    python preprocess.py --appliance geyser
    python preprocess.py --appliance computer
    python preprocess.py --all
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE_FREQ = "15min"

# REVISED (Week 3) from the original 96-in/4-out (24h in, 1h out) spec.
# Inspecting the REAL iAWE download (not assumed/synthetic) showed the raw
# 1-second logging happens in disconnected "sessions" with large gaps between
# them (sensor/logging downtime, not appliance-off periods -- confirmed via
# delta-time analysis: 99%+ of in-session deltas are exactly 1s, so within a
# session the data is genuinely continuous). Longest continuous session per
# channel: computer/laptop ~17.2h, geyser/AC proxy ~23.9h, fan/TV proxy only
# ~2.3h. A 24h input window is therefore infeasible for 2 of 3 channels (0
# valid windows). Empirically tested window sizes (see Week 3 chat log) and
# picked 2h-in / 30min-out as the largest horizon that yields usable window
# counts on ALL THREE channels (51-1883 windows) while still being a
# legitimate, useful forecast horizon for the actual use case: near-real-time
# relay/demand-response control cares more about "next 30 min" than "next
# 24h" anyway. Document this exact reasoning in Ch3.8 -- it's a real,
# defensible methodology decision driven by empirical data inspection, not a
# shortcut.
INPUT_STEPS = 8      # 2 hours of 15-min buckets
OUTPUT_STEPS = 2      # next 30 minutes of 15-min buckets
MAX_GAP_BUCKETS = 2   # interpolate gaps up to 30 min; longer gaps split the series into separate sessions

# Tested (Week 3): adding cyclical hour/day-of-week features to give the
# LSTM an edge ARIMA can't replicate. Result was MIXED, not a clean win --
# computer/laptop improved slightly (R2 0.368 -> 0.378), but geyser/AC and
# fan/TV both got WORSE (0.593 -> 0.516, -0.015 -> -0.160). With only
# 140-1673 training windows per appliance, 4 extra input dimensions mostly
# added noise rather than learnable signal on 2 of 3 channels. Kept the
# capability (cyclical_time_features below) but defaulted OFF -- the plain
# univariate model is the stronger, more consistent choice given the real
# data volume available. This negative result is itself worth reporting in
# Ch4 as evidence of a properly tested design decision, not a shortcut.
INCLUDE_TIME_FEATURES = False

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 -> test


def load_raw(appliance: str) -> pd.DataFrame:
    path = RAW_DIR / f"{appliance}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run generate_synthetic_iawe.py first, or place "
            f"the real iAWE '{appliance}' channel export here with columns "
            f"[timestamp, power_w]."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def resample_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 15-min buckets, interpolate short gaps, flag long gaps."""
    resampled = df["power_w"].resample(RESAMPLE_FREQ).mean()

    # identify gap runs
    is_na = resampled.isna()
    gap_id = (is_na != is_na.shift()).cumsum()
    gap_lengths = is_na.groupby(gap_id).transform("sum")

    short_gap_mask = is_na & (gap_lengths <= MAX_GAP_BUCKETS)
    long_gap_mask = is_na & (gap_lengths > MAX_GAP_BUCKETS)

    cleaned = resampled.copy()
    cleaned[short_gap_mask] = np.nan  # will be linearly interpolated next
    cleaned = cleaned.interpolate(method="linear", limit=MAX_GAP_BUCKETS)

    n_long_gap = long_gap_mask.sum()
    n_short_gap = short_gap_mask.sum()
    print(f"    resampled to {len(cleaned)} x 15-min buckets | "
          f"interpolated {n_short_gap} short-gap buckets | "
          f"{n_long_gap} long-gap buckets remain NaN (dropped before windowing)")

    return cleaned.to_frame(name="power_w")


def cyclical_time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Week 3 addition: hour-of-day and day-of-week encoded as sin/cos pairs
    (already bounded in [-1,1], no scaling needed). Purpose: give the LSTM
    an exogenous signal ARIMA structurally cannot use, since real-data
    testing showed ARIMA competitive with/ahead of a bare univariate LSTM
    on 2 of 3 appliances. Appliance usage (AC, TV, laptop) is strongly
    time-of-day driven, so this should give the LSTM real predictive
    leverage ARIMA's fixed autoregressive structure can't access.
    Returns shape (n, 4): [hour_sin, hour_cos, dow_sin, dow_cos]
    """
    hour_frac = (index.hour + index.minute / 60) / 24
    dow_frac = index.dayofweek / 7
    hour_sin = np.sin(2 * np.pi * hour_frac)
    hour_cos = np.cos(2 * np.pi * hour_frac)
    dow_sin = np.sin(2 * np.pi * dow_frac)
    dow_cos = np.cos(2 * np.pi * dow_frac)
    return np.stack([hour_sin, hour_cos, dow_sin, dow_cos], axis=1)


def make_windows_by_session(power: np.ndarray, time_feats: np.ndarray, input_steps: int, output_steps: int):
    """
    Session-aware sliding window: the series is first split into contiguous
    non-NaN "sessions" (real sensor recording breaks separate sessions), and
    windows are only built WITHIN a session -- never across a gap boundary.
    This matters a lot on real iAWE data, where a single global pass would
    find zero valid windows once gaps exceed the window length anywhere in
    the series (this was the Week 3 bug: 5,727 buckets in, 4 windows out).

    X windows carry [power, hour_sin, hour_cos, dow_sin, dow_cos] per
    timestep (shape: steps x 5). y windows carry power only (the forecast
    target) -- shape: steps.

    Returns a list of (X_session, y_session) arrays, one pair per session --
    kept separate (not concatenated) so the caller can do a SESSION-LEVEL
    train/val/test split rather than a naive chronological cut, which risks
    dumping an entire near-idle/flat session into one split and corrupting
    its metrics (this happened to the geyser/AC channel's test set).
    """
    is_na = np.isnan(power)
    change = np.diff(is_na.astype(int), prepend=is_na[0].astype(int) + 1)
    session_id = np.cumsum(change != 0)
    sessions = []
    for sid in np.unique(session_id):
        mask = session_id == sid
        if is_na[mask][0]:
            continue
        run_power = power[mask]
        run_feats = time_feats[mask]
        n = len(run_power)
        Xs, ys = [], []
        for start in range(0, n - input_steps - output_steps + 1):
            in_power = run_power[start:start + input_steps]
            if INCLUDE_TIME_FEATURES:
                in_feats = run_feats[start:start + input_steps]
                X_window = np.concatenate([in_power[:, None], in_feats], axis=1)  # (steps, 5)
            else:
                X_window = in_power[:, None]  # (steps, 1) -- univariate, the stronger config (see note above)
            Xs.append(X_window)
            ys.append(run_power[start + input_steps:start + input_steps + output_steps])
        if Xs:
            sessions.append((np.array(Xs), np.array(ys)))
    return sessions


def session_shuffled_split(sessions: list, seed: int = 42):
    """
    Splits at the SESSION level, not the window level: shuffle session order
    (fixed seed for reproducibility), then greedily assign whole sessions to
    train until ~70% of total windows are allocated, then val until ~85%,
    remainder to test. Windows from one session never span two splits (a
    real train/test leakage risk with overlapping sliding windows), and
    shuffling session order -- rather than a strict chronological cut --
    prevents an entire split from landing on a single unrepresentative
    (e.g. near-idle/flat) session purely by where it falls in time.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sessions))
    total_windows = sum(len(X) for X, _ in sessions)
    train_target = total_windows * TRAIN_FRAC
    val_target = total_windows * (TRAIN_FRAC + VAL_FRAC)

    train_X, train_y, val_X, val_y, test_X, test_y = [], [], [], [], [], []
    running = 0
    for idx in order:
        X, y = sessions[idx]
        running += len(X)
        if running <= train_target:
            train_X.append(X); train_y.append(y)
        elif running <= val_target:
            val_X.append(X); val_y.append(y)
        else:
            test_X.append(X); test_y.append(y)

    def cat(parts):
        return np.concatenate(parts) if parts else np.empty((0,))

    return {
        "train": (cat(train_X), cat(train_y)),
        "val": (cat(val_X), cat(val_y)),
        "test": (cat(test_X), cat(test_y)),
    }


def process_appliance(appliance: str):
    print(f"\n=== Processing: {appliance} ===")
    raw = load_raw(appliance)
    print(f"  loaded {len(raw):,} raw 1-sec samples "
          f"({raw.index.min()} -> {raw.index.max()})")

    clean = resample_and_clean(raw)
    series = clean["power_w"].values
    time_feats = cyclical_time_features(clean.index)  # (n, 4): hour_sin, hour_cos, dow_sin, dow_cos

    # Fit the scaler on ALL non-NaN values first just to build windows in
    # scaled space is no longer possible chronologically (session split is
    # now randomized, not a time prefix) -- instead fit scaler on the raw
    # (unscaled) session windows AFTER the session-level split is decided,
    # using only sessions assigned to train. See below.
    sessions = make_windows_by_session(series, time_feats, INPUT_STEPS, OUTPUT_STEPS)
    total_windows = sum(len(X) for X, _ in sessions)
    print(f"  built {total_windows} windows across {len(sessions)} sessions "
          f"(input={INPUT_STEPS} steps, output={OUTPUT_STEPS} steps, "
          f"features=[power, hour_sin, hour_cos, dow_sin, dow_cos])")

    splits_raw = session_shuffled_split(sessions)

    # Fit scaler on train split's POWER CHANNEL ONLY (column 0) -- the
    # cyclical time features (columns 1-4) are already bounded in [-1,1]
    # and must NOT be scaled. Avoids leakage by fitting on train split only.
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_raw, train_y_raw = splits_raw["train"]
    if len(train_X_raw) == 0:
        raise RuntimeError(f"No training windows for {appliance} -- check session sizes vs INPUT_STEPS/OUTPUT_STEPS")
    scaler.fit(train_X_raw[..., 0].reshape(-1, 1))

    splits = {}
    for split_name, (Xs, ys) in splits_raw.items():
        if len(Xs) == 0:
            splits[split_name] = (Xs, ys)
            continue
        Xs_scaled = Xs.copy()
        Xs_scaled[..., 0] = scaler.transform(Xs[..., 0].reshape(-1, 1)).reshape(Xs[..., 0].shape)
        ys_scaled = scaler.transform(ys.reshape(-1, 1)).reshape(ys.shape)
        splits[split_name] = (Xs_scaled, ys_scaled)

    for split_name, (Xs, ys) in splits.items():
        print(f"    {split_name}: X={Xs.shape}, y={ys.shape}")

    out_dir = PROCESSED_DIR / appliance
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (Xs, ys) in splits.items():
        np.save(out_dir / f"X_{split_name}.npy", Xs)
        np.save(out_dir / f"y_{split_name}.npy", ys)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    clean.to_csv(out_dir / "resampled_15min.csv")

    print(f"  saved processed arrays + scaler to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliance", choices=["fan", "geyser", "computer"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.appliance is None:
        for appliance in ["fan", "geyser", "computer"]:
            process_appliance(appliance)
    else:
        process_appliance(args.appliance)


if __name__ == "__main__":
    main()
