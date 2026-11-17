# Fixed threshold vs LSTM forecast (offline test, 1-hour models)

Script: `scripts/compare_fixed_vs_lstm.py` (run in `scripts/`: `python compare_fixed_vs_lstm.py`)
Full numbers: `results_1h/fixed_vs_lstm.csv`

## What was tested
Held-out iAWE test windows (never used in training). Three ways to decide, same limit:
- **FIXED**: act if the reading NOW is above the limit (a normal relay system).
- **LSTM**: act if the 30-min forecast is above the limit (this project).
- **BOTH**: act if either is true (tested as an option, not in the live system yet).

Truth = did the real power in the next 30 min go above the limit?
Repeated at 3 limit levels: normal (x1.00), near a TNB bracket (x0.85), at the bracket (x0.70).

Terms: *caught* = high load came and the method acted. *missed* = high load came, no action.
*false alarm* = acted but no high load came. *early catch* = acted while the reading was still
below the limit (a fixed threshold can never do this). *stale alarm* = acted on a high reading
that was about to drop.

## Key results

| Appliance, limit | Method | Caught | Missed | False alarms | Early catches |
|---|---|---|---|---|---|
| Fan, near bracket (72 W) | FIXED | 59 / 71 | 12 | 6 | 0 |
| Fan, near bracket (72 W) | **LSTM** | **71 / 71** | **0** | 8 | **12** |
| Heater, at bracket (1330 W) | FIXED | 58 / 65 | 7 | 2 | 0 |
| Heater, at bracket (1330 W) | **LSTM** | **60 / 65** | 5 | 2 | **3** |
| Heater, near bracket (1615 W) | **FIXED** | **50 / 57** | 7 | 2 | 0 |
| Heater, near bracket (1615 W) | LSTM | 43 / 57 | 14 | 3 | 2 |
| Laptop, at bracket (50 W) | FIXED | 40 / 66 | 26 | 15 | 0 |
| Laptop, at bracket (50 W) | **LSTM** | 41 / 66 | 25 | **4** | **7** |
| Laptop, normal (72 W) | **FIXED** | **11 / 23** | 12 | 9 | 0 |
| Laptop, normal (72 W) | LSTM | 1 / 23 | 22 | 1 | 1 |

(At the normal x1.00 limit the heater and fan test data almost never go above the limit, so there is nothing to compare there.)

## Plain-words conclusion (honest)
- **Where the LSTM wins:** it catches some high loads *before* they happen (early catches: fan 12, heater 3, laptop 7), which a fixed threshold cannot do at all. On the fan near a bracket it caught all 71 vs 59. On the laptop it raised far fewer needless alarms (4 vs 15), because it does not react to short bursts that are about to drop.
- **Where the fixed threshold wins:** the LSTM forecast is smoother than reality, so it under-predicts some peaks. Heater near the bracket: fixed caught 50, LSTM 43. Laptop at the normal limit: the LSTM almost never fires (1 of 23).
- **So:** the LSTM is not better everywhere. It is better at acting *early* and at *not over-reacting*; a fixed threshold is better at catching a load that is already high and stays high.
- **Limits of this test:** small test sets (heater 97, fan 81, laptop 452 windows), proxy channels (heater = AC, fan = TV), offline data from iAWE (Delhi 2013), not live hardware. Report the numbers as indicative.
- **Using both together works best on catching:** acting if the reading now OR the forecast is above the limit caught the most high loads in every case (heater near bracket 52/57, heater at bracket 61/65, fan 71/71, laptop at bracket 47/66), with a few more alarms than either alone. This is only tested offline; the live system still uses the forecast alone.
