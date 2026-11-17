"""
Week 3 correction: replaces the synthetic placeholder data with the REAL
iAWE dataset (electricity.tar.gz, downloaded by Ilyasa/Aizziq directly from
iawe.github.io -- no request form needed, contrary to what was assumed in
Week 1 planning).

IMPORTANT SCOPE CORRECTION vs. earlier assumption:
    iAWE's actual labels.dat (verified from the real download, not assumed):
        1 mains, 2 mains, 3 fridge, 4 air conditioner, 5 air conditioner,
        6 washing machine, 7 laptop computer, 8 iron, 9 kitchen outlets,
        10 television, 11 water filter, 12 water motor
    There is NO "geyser" (water heater) channel and NO "fan" channel in this
    iAWE release -- that was an incorrect assumption made before the real
    file was inspected. Substitutions below are chosen empirically (by
    actual power profile, not by name) and documented for Ch3.8 defense:

    - laptop        -> meter 7  "laptop computer"   EXACT MATCH, no substitution needed.
                        2.55M samples, ~59 days coverage.

    - water heater   -> meter 4  "air conditioner"   SUBSTITUTED.
                        Reasoning: the only channels resembling a resistive
                        thermostat-cycled heating load are "iron" (meter 8,
                        only 7,134 samples =~ 2 hours -- nowhere near enough
                        for LSTM training, which needs 96+ step windows) and
                        "kitchen outlets" (meter 9, only 13,340 samples =~
                        3.7 hours -- same problem). Air conditioner (meter 4)
                        is the best-populated high-power, thermostatically
                        cycled load available (640K samples =~ 7.4 days,
                        mean-when-active 1595W, comparable magnitude to a
                        2000-3000W water heater, and it duty-cycles on/off
                        to hold a setpoint the same way a water heater does).
                        This is a data-availability substitution, not a
                        name-based one -- document it exactly this way in
                        the report, don't paper over it.

    - kipas diri     -> meter 10 "television"        SUBSTITUTED.
                        Reasoning: no fan channel exists. Television is the
                        best-matched available appliance by wattage (73.6W,
                        vs. kipas diri's typical 30-70W) and by usage
                        pattern (continuous draw while on, user-toggled
                        on/off rather than thermostat-controlled, similar to
                        how a fan is used). 428,937 samples =~ 5 days.

Usage:
    python extract_real_iawe.py --iawe-dir /path/to/extracted/electricity
"""

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# internal pipeline key -> (iAWE meter number, iAWE label, real vs substituted)
CHANNEL_MAP = {
    "computer": (7, "laptop computer", "EXACT MATCH"),
    "geyser": (4, "air conditioner", "SUBSTITUTED (water heater proxy)"),
    "fan": (10, "television", "SUBSTITUTED (kipas diri proxy)"),
}


def extract(iawe_electricity_dir: Path):
    for key, (meter, label, status) in CHANNEL_MAP.items():
        src = iawe_electricity_dir / f"{meter}.csv"
        if not src.exists():
            raise FileNotFoundError(f"Expected {src} -- did you extract electricity.tar.gz?")

        df = pd.read_csv(src, usecols=["timestamp", "W"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.rename(columns={"W": "power_w"})
        df = df[["timestamp", "power_w"]].sort_values("timestamp")

        out_path = RAW_DIR / f"{key}.csv"
        df.to_csv(out_path, index=False)
        print(f"  {key:9s} <- iAWE meter {meter:>2} ({label:<16s}) [{status}] "
              f"| {len(df):,} rows | {df['timestamp'].min()} -> {df['timestamp'].max()} "
              f"| saved -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iawe-dir", type=Path, required=True,
                         help="Path to the extracted 'electricity' folder from iAWE's electricity.tar.gz")
    args = parser.parse_args()

    print("Extracting real iAWE channels (replacing synthetic placeholders in data/raw/):")
    extract(args.iawe_dir)
    print("\nDone. Run preprocess.py --all next to rebuild windows on real data.")


if __name__ == "__main__":
    main()
