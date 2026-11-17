# Setup and Test Guide

Everything below has already been run once in the delivery environment and
confirmed working — this is the reproducible version so you (or Ilyasa) can
run it again on another machine, verify it, or pick up from any step.

There are two independent parts: the **Python pipeline** (data → LSTM/ARIMA →
control logic — runs on a laptop/Colab, no hardware needed) and the
**firmware** (ESP32 + relay — needs the physical hardware). Do Part A first;
Part B only makes sense once Part A produces working models.

---

## Part A — Python pipeline

### A0. Prerequisites

- Python 3.10+ (check with `python3 --version`)
- The `ilyasa_lstm/` folder, complete with `scripts/`, `data/raw/*.csv` (the
  real iAWE channels already extracted), `models/`, `results/`

### A1. Install dependencies

```bash
cd ilyasa_lstm
pip install -r requirements.txt --break-system-packages
```

Expect this to take a few minutes — TensorFlow is the largest package.
Verify it worked:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
python3 -c "import statsmodels; import sklearn; import matplotlib; print('OK')"
```

You should see a TensorFlow version number (e.g. `2.21.0`) and `OK`, with
no `ModuleNotFoundError`.

### A2. (Only if starting from scratch) Extract real iAWE data

If `data/raw/fan.csv`, `geyser.csv`, `computer.csv` already exist, skip this
step — they're already built from the real iAWE download. If starting fresh
with a new `electricity.tar.gz` from https://iawe.github.io/:

```bash
tar -xzf electricity.tar.gz    # produces an `electricity/` folder
cd scripts
python3 extract_real_iawe.py --iawe-dir /path/to/electricity
```

Expected output: three lines confirming meter 7 (laptop, exact match),
meter 4 (air conditioner → water heater proxy), meter 10 (television → fan
proxy), each with row counts and date ranges.

### A3. Run preprocessing

```bash
cd scripts
python3 preprocess.py --all
```

Expected output (roughly): fan → 208 windows across 44 sessions, geyser →
355 windows across 43 sessions, computer → 2,392 windows across 87 sessions.
Each appliance prints train/val/test shapes like `train: X=(1673, 8, 1)`.

If you see `built 0 windows` or `built 4 windows` for any appliance, something regressed — check that `INPUT_STEPS`/`OUTPUT_STEPS` in `preprocess.py` are still `8`/`2` (not reverted to the original `96`/`4`, which is too large for this dataset's real session lengths).

### A4. Train the LSTM models

```bash
python3 train_lstm.py --all
```

Takes about 1-2 minutes total on CPU. Expected final summary (numbers will
vary slightly run-to-run due to random initialization, but should be in
this ballpark):

```
fan        RMSE~1.0W   R2~-0.1 to -0.2   (weak — thin, near-constant data, expected)
geyser     RMSE~450-550W  R2~0.55-0.65   (this is the strongest result — cite this one)
computer   RMSE~12-14W    R2~0.35-0.40
```

If `geyser` stops after fewer than ~30 epochs, check `PATIENCE = 20` in
`train_lstm.py` wasn't reverted — an earlier bug had this at `8`, which cut
training short and produced much worse results.

### A5. Train the ARIMA baselines

```bash
python3 train_arima.py --all
```

Takes under 30 seconds. Expected: fan R2 strongly negative (both models are
weak here, ARIMA even more so), geyser R2~0.43, computer R2~0.62 (ARIMA
currently beats LSTM on laptop — this is a real, reported result, not a bug).

### A6. Compile the comparison table

```bash
python3 compile_results.py
```

Prints a comparison table plus a set of caveats about how LSTM/ARIMA are
evaluated on different held-out samples — read those caveats before quoting
any RMSE number directly against the other model. Produces
`results/comparison_table.csv` and `results/lstm_vs_arima_rmse.png`.

### A7. Sanity-check the control logic (no data needed, pure logic test)

```bash
python3 control_logic.py
```

Runs 7 scenarios and prints the decision for each (the first 4 are the
original proxy-scale regression checks; the last 3 use the real physical
demo hardware's actual wattage — 1500W water heater, 70W fan — confirmed
Week 4). Current expected output:

```
geyser     cur=  1200W pred=  1250W cum_month=  150kWh -> action=OK       tier=yellow threshold=1620W
geyser     cur=  1200W pred=  1950W cum_month=  190kWh -> action=CUTOFF   tier=red    threshold=1377W
fan        cur=    60W pred=    70W cum_month=  100kWh -> action=OK       tier=yellow threshold=85W
computer   cur=    40W pred=    58W cum_month=   50kWh -> action=OK       tier=yellow threshold=72W
geyser     cur=  1500W pred=  1500W cum_month=  150kWh -> action=NOTIFY   tier=orange threshold=1620W
geyser     cur=  1500W pred=  1500W cum_month=  195kWh -> action=CUTOFF   tier=red    threshold=1255W
fan        cur=    70W pred=    70W cum_month=  150kWh -> action=OK       tier=yellow threshold=85W
```

The geyser `threshold` was rescaled from 1900W to 1620W in Week 4 to match
the real 1500W water heater Ilyasa is actually using (see the comment block
above `APPLIANCE_THRESHOLDS_W` in `control_logic.py` for the exact
rationale) — if you see `threshold=1900W` anywhere, the rescale didn't take.
The two 1500W scenarios are the ones worth watching: at 150kWh (far from a
tariff boundary) the real heater running at its own normal full power lands
in `NOTIFY`/orange, not `CUTOFF` — that's intentional, not a bug. At 195kWh
(5kWh from the 200kWh boundary) the exact same appliance, unchanged, tips
into `CUTOFF`/red purely from the dynamic tariff-tier tightening — that's
the "predictive + tariff-aware" behaviour actually doing something
observable, and it's the pair of rows worth pointing at in a demo or viva
if asked to show the dynamic threshold working on the real hardware. If the
first geyser CUTOFF case (row 2, proxy-scale) comes back as anything other
than `CUTOFF`, the appliance policy logic has been altered — check the
`if appliance == "geyser"` branch in `decide()`.

### A8. Run the full control simulation against real predictions

```bash
python3 run_control_simulation.py --all
```

This loads each trained model, runs it against real held-out test windows,
and pushes every prediction through the control logic — the actual
end-to-end pipeline. Produces `results/<appliance>_control_simulation.csv`
(20 rows each) and `results/control_simulation_summary.json`. Open the
geyser CSV and confirm `threshold_w` decreases as `cumulative_kwh`
increases toward 200 — that's the dynamic tariff-tier behaviour working.

**A1 through A8 should take under 5 minutes total** on a normal laptop. If
everything above ran without errors, Part A is fully verified.

---

## Part B — Firmware (needs physical hardware)

Full detail — wiring diagram, safety warnings, mains-voltage handling — is
in `firmware/BUILD_GUIDE.md`. Summary of the steps, in order:

1. **Arduino IDE libraries needed** (install via Library Manager if not
   already present from the base PSM1 prototype): `Blynk` (search
   "Blynk" by Volodymyr Shymanskyy), `LiquidCrystal I2C` (by Frank de
   Brabander or Marco Schwartz — either works with the `LiquidCrystal_I2C`
   include used in the sketch), ESP32 board support (already required for
   the existing PZEM-004T sketch).
2. **Wire the logic side only** (ESP32 ↔ relay module IN/VCC/GND, ESP32 ↔
   buzzer) — this is low-voltage, safe to do without any mains involved.
   See `BUILD_GUIDE.md` for exact pins.
3. **Set up Blynk virtual pins**: `V10` (int, control value 0-3), `V11`
   (string, status text), `V12` (read-only relay state feedback).
4. **Fill in credentials** at the top of `relay_control.ino` (Blynk auth
   token, WiFi SSID/password) and flash to the ESP32.
5. **Test on a SAFE load first** — a lamp or the bulb circuit from the
   existing Ch4 prototype, NOT the real water heater yet. From the Blynk
   app, manually write `0`, `1`, then `3` to `V10` and confirm the lamp
   and LCD respond as described in `BUILD_GUIDE.md`'s testing section.
6. **Test the fail-safe**: disconnect WiFi for 5+ minutes, confirm the
   relay defaults to ON and the LCD shows "FAILSAFE MODE".
7. **Only after all of the above passes**, move to the mains-voltage
   wiring for the real water heater, following the safety procedure in
   `BUILD_GUIDE.md` exactly (breaker off, verify dead with a tester,
   insulate all connections before restoring power).

## Wiring the two parts together (Python → Blynk → Firmware)

`run_control_simulation.py` still runs the LSTM and control logic against
**saved test-set data**, not live sensor readings — keep using it for
offline validation. The live version now exists: `scripts/live_control_loop.py`.

It talks to Blynk over plain HTTPS (confirmed against Blynk's real API docs,
not guessed) — never to the ESP32 directly:

- `GET https://blynk.cloud/external/api/get?token=...&{pin}` — polls the
  current PZEM reading every 15 minutes.
- `GET https://blynk.cloud/external/api/update?token=...&{pin}={value}` —
  pushes the decision back to `V10`/`V11`, which the ESP32's existing
  `BLYNK_WRITE(V_CONTROL)` handler in `relay_control.ino` picks up and acts
  on (same as if you'd written it by hand in the Blynk app).

It builds its own local rolling 2-hour window (`data/live/<appliance>_buffer.csv`)
rather than depending on Blynk's history API, so there's a roughly 2-hour
warm-up period after first starting it before predictions begin — this is
expected, not a bug (the script logs "warming up" during this window).

**Before running it for real, fill in / confirm two things at the top of
`live_control_loop.py`:**
1. `BLYNK_TOKEN` — same device auth token already in `relay_control.ino`.
2. `POWER_PIN` — whichever virtual pin the base PSM1 sketch actually posts
   the PZEM's live power reading to. This isn't confirmed against Ilyasa's
   real firmware (only the PSM1 report PDF was available here, which
   doesn't include the sketch's pin table) — placeholder is `"V0"`.

`CURRENT_APPLIANCE` at the top is a manual switch ("geyser" or "fan") —
change it to match whichever appliance is physically plugged into the
socket box before starting a session, since there's one PZEM shared
sequentially between them, not three running at once.

Run it with:
```bash
python3 live_control_loop.py           # runs continuously, polling every 15 min
python3 live_control_loop.py --once    # single cycle, useful for testing or cron
```

One documented limitation worth repeating to the client: `cumulative_kwh`
is tracked from the ONE monitored appliance's own usage as a stand-in for
household-level TNB tier proximity, not read from a real household smart
meter — set `MANUAL_CUMULATIVE_KWH` in the script if Ilyasa wants to plug
in his actual meter reading instead of the auto-tracked proxy.

---

## Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError` on any Python step | Re-run A1, add `--break-system-packages` if using system Python |
| `preprocess.py` reports 0 windows | `INPUT_STEPS`/`OUTPUT_STEPS` reverted to 96/4 — check they're 8/2 |
| Geyser R² much worse than ~0.6 | `PATIENCE` in `train_lstm.py` reverted below 20 |
| Fan control simulation shows `SHIFT` on every single row | Expected in a low-usage-margin test window (fan/TV is near-constant power, see Ch3.9.2) — not itself a bug, but if it happens even when `cumulative_kwh` starts well below 180 in `run_control_simulation.py`, check `APPLIANCE_THRESHOLDS_W["fan"]` wasn't reverted to the earlier miscalibrated 65W |
| Relay doesn't respond to Blynk writes | Confirm virtual pin numbers match (`V10`/`V11`/`V12`) between the Blynk app widgets and the `#define` lines in `relay_control.ino` |
