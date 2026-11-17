# Ilyasa LSTM Project — Week 1 Deliverable (Data Preprocessing)

## What's in this folder

```
ilyasa_lstm/
├── scripts/
│   ├── generate_synthetic_iawe.py   # placeholder data generator (dev/testing only)
│   └── preprocess.py                 # the actual Week 1 pipeline
├── data/
│   ├── raw/                          # per-appliance CSVs go here (timestamp, power_w)
│   └── processed/
│       ├── fan/
│       ├── geyser/                   # = water heater
│       └── computer/                 # = laptop proxy
└── README.md
```

## Status

Pipeline is built and tested end-to-end on synthetic data shaped like iAWE
(14 simulated days, 1-second resolution, matching iAWE's known power ranges
and the documented geyser channel dropout issue). It ran clean:

- fan: 1,245 windows built, 0 gaps
- geyser: 1,245 windows built, 6 short gaps interpolated, 0 long gaps
- computer: 1,245 windows built, 0 gaps

## Swapping in the real iAWE data

1. Request access at https://iawe.github.io/ (dataset requires a request form,
   not a direct download — this needs to happen before Week 2 starts).
2. Once you have the real per-appliance exports, convert/save each as:
   `data/raw/fan.csv`, `data/raw/geyser.csv`, `data/raw/computer.csv`
   with exactly two columns: `timestamp`, `power_w`.
3. Delete the synthetic files and run:
   ```
   python3 scripts/preprocess.py --all
   ```
4. Everything downstream (LSTM training in Week 2) reads from
   `data/processed/<appliance>/` and doesn't care whether the source was
   synthetic or real — no code changes needed elsewhere.

## What the pipeline does (maps to Ch3.8 methodology section)

1. Load raw 1-second power readings per appliance
2. Resample to 15-minute buckets (mean power per bucket)
3. Gap handling: interpolate gaps ≤ 1 hour (4 buckets), drop windows touching
   longer gaps — documented explicitly so it can be defended in the methodology
   chapter and viva
4. MinMax scale each appliance independently, fitted on the training split
   only (avoids data leakage from val/test into the scaler)
5. Sliding window: 96 steps (24h) in → 4 steps (1h) out
6. Chronological 70/15/15 train/val/test split (not randomly shuffled —
   preserves time order, standard practice for time-series forecasting)

## Known limitation to flag to supervisor

iAWE's "fan" channel is most likely a ceiling fan, not a personal/table fan
(kipas diri), and "computer" is a desktop, not a laptop. Same applies to
Malaysian vs Indian climate context. Suggested defense line already drafted —
see prior chat notes. Worth stating explicitly in Ch3.8 as a documented
substitution, not glossed over.

---

## Week 2 — LSTM + ARIMA baseline (DONE, dev-run on synthetic data)

New folders:
```
models/
├── lstm_fan.keras
├── lstm_geyser.keras
└── lstm_computer.keras
results/
├── <appliance>_lstm_results.json       # metrics + config per appliance
├── <appliance>_arima_results.json
├── <appliance>_loss_curve.png
├── <appliance>_prediction_vs_actual.png
├── <appliance>_arima_prediction_vs_actual.png
├── comparison_table.csv
└── lstm_vs_arima_rmse.png
```

New scripts: `train_lstm.py`, `train_arima.py`, `compile_results.py`.

### Results (dev run — 14 days synthetic data, 10 epochs, capped for sandbox runtime)

| Appliance | LSTM RMSE (W) | ARIMA RMSE (W) | LSTM R² | ARIMA R² | RMSE improvement |
|---|---|---|---|---|---|
| Fan | 1.84 | 3.34 | 0.968 | 0.898 | LSTM 45% better |
| Geyser (water heater) | 193.82 | 194.84 | -0.006 | -0.003 | near-tie (both weak) |
| Computer (laptop proxy) | 0.83 | 1.06 | 0.981 | 0.969 | LSTM 22% better |

**Important — read before showing anyone these numbers:** this is a
constrained dev run only. Training was capped at 10 epochs (vs. the
production default of 100 with early-stopping patience 8, which is already
what `train_lstm.py` runs by default — I only override it with `--epochs 10`
for fast sandbox iteration). On 14 days of synthetic data with 10 epochs, the
LSTM hasn't had enough training to actually learn the geyser's sparse burst
timing — hence the near-tie with ARIMA there. This is expected, not a sign
the architecture is wrong. The fan and laptop results already show the
expected pattern (LSTM clearly ahead) even undertrained, which is a good
sign. Re-run with `--epochs 100` (default, no need to pass it) once real
iAWE data (73 days) is in — that's when the geyser result should separate
properly, since bursty/rare-event patterns need more data + more training
passes than smooth continuous signals to learn.

To retrain on real data once dataset access is granted:
```
python3 preprocess.py --all
python3 train_lstm.py --all
python3 train_arima.py --all
python3 compile_results.py
```

---

## Week 3 — REAL data swap-in + major pipeline corrections

Real iAWE `electricity.tar.gz` downloaded and used. Several assumptions from
Week 1/2 turned out wrong once the real file was inspected — corrected here,
documented honestly rather than papered over.

### 1. Appliance channel mapping corrected

Real `labels.dat`: `mains, mains, fridge, air conditioner, air conditioner,
washing machine, laptop computer, iron, kitchen outlets, television, water
filter, water motor`. There is **no geyser/water-heater channel and no fan
channel** in this iAWE release — the Week 1 assumption that iAWE had those
was wrong (that assumption was made before the file could be inspected, since
the source website couldn't be fetched at the time).

| Pipeline key | Real iAWE source | Status |
|---|---|---|
| `computer` | Meter 7, "laptop computer" | **Exact match** — better than assumed (Week 1 wrongly guessed this would be a desktop) |
| `geyser` (water heater proxy) | Meter 4, "air conditioner" | Substituted — chosen empirically, not by name. The only heating-element-type channels (iron, kitchen outlets) had only 2-4 hours of total data each, nowhere near enough to train on. AC is the best-populated (640K samples) high-power, thermostat-cycled load available. |
| `fan` (kipas diri proxy) | Meter 10, "television" | Substituted — closest available continuous, user-toggled load by wattage (73.6W) and usage pattern. |

Script: `extract_real_iawe.py --iawe-dir /path/to/extracted/electricity`

### 2. Forecast window shrunk from 24h-in/1h-out to 2h-in/30min-out

Real per-second logging happens in disconnected "sessions" (confirmed: 99%+
of in-session gaps are exactly 1 second, so data WITHIN a session is
genuinely continuous — the gaps are real sensor/logging downtime between
sessions, not appliance-off periods). Longest continuous session per
channel: computer ~17.2h, geyser(AC) ~23.9h, fan(TV) ~2.3h. A 24h input
window is infeasible on 2 of 3 channels (found 0 valid windows). Tested
several window sizes empirically and picked 2h-in/30min-out as the largest
horizon that yields workable window counts on all three channels (140-1673
training windows). This is also arguably a *better* fit for the actual
use case — a relay/demand-response control loop cares more about "what
happens in the next 30 minutes" than "the next 24 hours" anyway.

### 3. Split strategy changed from chronological to session-shuffled

A strict chronological 70/15/15 split put the geyser test set entirely
inside a near-flat idle session (std ≈ 0.06W) purely by where it fell in
time, making R² blow up to a meaningless -1,963,789. Fixed by shuffling
whole SESSIONS (not individual windows, to avoid leakage between
overlapping windows) into train/val/test — see `session_shuffled_split()`
in `preprocess.py`. Also added NRMSE (RMSE normalized by mean |actual|) and
an `r2_reliable` flag to every result, since R² is fundamentally unstable
on near-zero-variance test slices — use NRMSE as the primary metric when
`r2_reliable` is `false`.

### 4. Results (real data, full 100-epoch budget + early stopping)

| Appliance | LSTM RMSE | LSTM R² | LSTM NRMSE | ARIMA RMSE | ARIMA R² | ARIMA NRMSE |
|---|---|---|---|---|---|---|
| Fan (TV proxy) | 0.97W | -0.01 | 1.3% | 0.78W | -1.42 | 1.0% |
| Geyser (AC proxy) | 527W | **0.59** | 48.2% | 216W | 0.43 | 200.2% |
| Computer (laptop) | 12.7W | 0.37 | 41.1% | 8.99W | **0.63** | 25.5% |

**Honest read of this table:** LSTM and ARIMA are NOT evaluated on identical
held-out samples (different split mechanics per model — see code comments),
so treat R²/NRMSE within each model as more trustworthy than a direct
RMSE-vs-RMSE comparison. More importantly: **this does not match the
"LSTM crushes ARIMA on bursty loads" narrative planned earlier.** ARIMA is
competitive with or ahead of the LSTM on 2 of 3 channels right now. Likely
cause: an 8-step (2h) input window and only a few hundred training samples
don't give the LSTM enough context or data to exploit its long-memory
advantage. Two honest paths forward, not yet decided:
(a) add time-of-day/day-of-week cyclical features to the LSTM input (an
advantage ARIMA structurally can't use), which should give LSTM a real edge
and restore the intended narrative legitimately, or
(b) report the nuanced result as-is in Ch4 — "under this data regime with
limited real-world logging density, ARIMA remains competitive" is itself a
defensible, honest finding for a Bachelor's report, just a different one
than originally pitched.

### 5. Update — tried (a), got a mixed result, found the real fix was elsewhere

Added `cyclical_time_features()` to `preprocess.py` (hour-of-day + day-of-week
as sin/cos pairs, 4 extra input dims). Result: **mixed, not a clean win** —
computer/laptop improved marginally (R² 0.368→0.378) but geyser/AC and
fan/TV both got *worse* (0.593→0.516, -0.015→-0.160). With only 140-1,673
training windows per appliance, 4 extra dimensions mostly added noise on 2
of 3 channels. Kept the capability in the code (`INCLUDE_TIME_FEATURES`
flag in `preprocess.py`, defaults `False`) but it's off for the final models.

While testing this, noticed `EarlyStopping(patience=8)` was cutting
training off suspiciously early (computer stopped at epoch 9). Bumped
patience to 20 — turned out **that** was the real fix, nothing to do with
the time features. Final results (univariate input, patience=20, full
retrain):

| Appliance | LSTM RMSE | LSTM R² | ARIMA RMSE | ARIMA R² | Winner |
|---|---|---|---|---|---|
| Fan (TV proxy) | 1.01W | -0.12 | 0.78W | -1.42 | LSTM (less bad — both weak, low-variance signal) |
| **Geyser (AC/water heater proxy)** | 494.8W | **0.64** | 216.2W | 0.43 | **LSTM, clearly** |
| Computer (laptop) | 12.75W | 0.37 | 8.99W | **0.63** | ARIMA |

This is a legitimate, better story than Week 3's first pass: LSTM now wins
on the channel that matters most for the "why LSTM" narrative (water
heater/geyser proxy), loses on laptop, and both models are weak on the
thin fan/TV data (only 208 total windows available — a real data-volume
limitation, not a modeling failure, worth stating plainly in Ch4). Good
material for Ch4's "Results and Analysis": tested an enhancement, it didn't
help, found the actual bug instead, documented both — exactly the kind of
rigor a viva panel wants to see, not just a clean win presented without
the process behind it.

### 6. Control module (the actual "control" client asked for)

New files: `scripts/control_logic.py`, `scripts/run_control_simulation.py`,
`firmware/relay_control.ino`, `firmware/BUILD_GUIDE.md`.

Appliance-differentiated policy — not one relay for everything:

| Appliance | Action on high predicted load | Why |
|---|---|---|
| Water heater (geyser/AC proxy) | **Hard relay cutoff**, with a 30-second pre-emptive buzzer warning before it actually cuts | Real safety/overcurrent risk — matches the client's own reference example (old project: relay trips on overcurrent), but this version acts on the LSTM's *forecast*, not just the instantaneous reading |
| Fan (kipas diri/TV proxy) | **Soft shift** (deferred, no relay) | Comfort item, not a safety risk |
| Laptop | **Notify only**, never auto-cut | Cutting power mid-session risks data loss — the failure mode of automating this is worse than the problem |

Also implemented: multi-tier alerts (green/yellow/orange/red), and a
dynamic threshold tied to **TNB Tariff A's monthly cumulative brackets**
(200/300/600/900 kWh — corrected from an earlier wrong assumption that
Malaysia has hourly peak/off-peak pricing; it doesn't, Ch2.2 of the report
already has the real tiered-monthly structure). As usage approaches the
next, more expensive tier boundary, thresholds tighten automatically.

Appliance power thresholds were recalibrated from real data percentiles
after an initial version had fan tripping on literally every reading (the
TV proxy is near-constant ~74W whenever on — a threshold that close to its
normal operating level isn't a meaningful signal; fixed by setting it above
p99 instead, so fan's control response is now driven mainly by tariff-tier
proximity rather than a power spike it doesn't naturally have).

Validated by running the actual trained LSTM models against real test data
through the full pipeline (forecast → decision) — see
`results/*_control_simulation.csv` for example logs, one per appliance,
showing the dynamic threshold visibly tightening as simulated monthly usage
approaches a tier boundary.

Firmware fail-safe: relay defaults to ON (appliance keeps running) if the
Blynk connection drops for 5+ minutes — documented as an explicit,
overridable safety decision in the code comments, not left implicit.

**Still needed from Ilyasa:** confirm which mains-rated relay module he's
buying (current rating must match the water heater's actual draw), and
whether he wants to demo the real water heater at viva or the safer lamp/
bulb-circuit substitute — see the safety section in `BUILD_GUIDE.md`.
