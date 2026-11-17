# HANDOVER: Ilyasa PSM2 – Smart Home Energy Demand Forecasting Using LSTM

Paste this file into the Claude Project (project knowledge) together with the key code files listed in section 4.
Freelancer: Aizziq. Client: Ilyasa (UTeM PSM2). Last updated: 27 Oct 2026.

## 1. What the project is
ESP32 measures one appliance with a PZEM-004T and can switch it with a relay. Readings go to Blynk IoT.
A Python script on the laptop reads Blynk, runs a trained LSTM to forecast the next 30 min of power,
decides OK / NOTIFY / SHIFT / CUTOFF, and writes the decision back to Blynk. The ESP32 then acts:
relay, buzzer, 16x2 LCD. Pricing uses TNB tariff (Time-of-Use or normal). A Streamlit dashboard shows everything.

Data path: PZEM -> ESP32 -> WiFi -> Blynk cloud <-> Python LSTM (laptop) -> Blynk -> ESP32 -> relay/buzzer/LCD.
The laptop never talks to the ESP32 directly.

## 2. Hardware (current, final scope)
- One PZEM-004T v3 (100A, CT on LIVE only) + one 5V relay module (30A), one socket box shared one appliance at a time:
  3W LED lamp (test load), fan (measured ~50W), 1500W water heater. Laptop has no live hardware.
- ESP32 DevKit (board "ESP32 Dev Module"), 16x2 I2C LCD (0x27, alt 0x3F), DHT22, active buzzer.
- Pins: relay IN GPIO26, buzzer GPIO27, PZEM TX -> GPIO16 (RX2), PZEM RX <- GPIO17 (TX2),
  LCD SDA GPIO21 / SCL GPIO22, DHT22 data GPIO4 (3.3V; 10k pull-up if bare sensor).
  Relay, PZEM, LCD VCC from VIN/5V. Common GND.
- Mains: plug L (10A fuse) -> through CT -> relay COM; relay NO -> socket L; N and E straight through;
  PZEM voltage terminals tapped from L and N before the relay. RCCB-protected wall socket.
- OPEN ISSUE: relay board is 5V active-low; with 3.3V IN it stays ON. Fix either (a) JD-VCC jumper: JD-VCC to 5V,
  VCC to 3.3V, set RELAY_ACTIVE_HIGH=false; or (b) RELAY_OPEN_DRAIN=true in the sketch. Test with firmware/pin_test (MODE 2).
  User had not confirmed the fix yet.

## 3. Blynk setup (done)
Template "Ilyasa Energy", Template ID TMPL6a5gyryUK, device "Ilyasa Energy", server sgp1.blynk.cloud.
Auth token: NOT stored here, copy from Blynk > Devices > Ilyasa Energy > Device Info.
Datastreams (16): V0 power_w | V3 energy kWh | V5 15-min avg W | V6 temp C | V7 humidity % |
V10 control 0-3 (OK/NOTIFY/SHIFT/CUTOFF) | V11 reason text | V12 relay state | V13 manual relay 0 AUTO/1 ON/2 OFF |
V14 buzzer 0 AUTO/1 ON/2 MUTE | V15 wattage mapping 0/1 | V16 ToU period text | V17 price sen/kWh |
V18 cost RM/hour | V19 tariff 1=ToU 0=normal | V20 cost today RM.
History is "1 Minute AVG" on V0, V3, V5, V6, V7, V17, V18, V20 (was "Last Value", which is why charts were empty).
Dashboard has gauges/labels/switches for all of the above, Power history chart (V0), Pricing history chart (V17+V18).
Controls only work on the Devices page, not the template page.

## 4. File directory (J:\UM_STUFF_2024\AFC Assignments\Assignments - Machine Learning\July2026\Ilyasa_LSTM\ilyasa_lstm\)
```
ilyasa_lstm/
  HANDOVER.md                  this file
  HOW_TO_RUN_UPDATES.md        short run instructions for the client
  README.md, SETUP_AND_TEST.md overall setup + expected test outputs
  requirements.txt             pandas numpy scikit-learn tensorflow statsmodels matplotlib requests streamlit
  Ch3_ML_Control_Writeup.docx  Ch3 draft (ML + control). NOTE: Ch2.2 tariff numbers are pre-July-2025, need update
  firmware/
    ilyasa_esp32_main/ilyasa_esp32_main.ino   MAIN ESP32 SKETCH (use this)
    pin_test/pin_test.ino                     relay/buzzer pin test (MODE 1 normal, 2 open-drain)
    relay_control.ino                         superseded, do not use
    Wiring_Diagram.svg, BUILD_GUIDE.md        wiring + socket-box build guide
  scripts/
    preprocess.py, train_lstm.py              2-hour-window LSTM (INPUT_STEPS=8)
    train_lstm_1h.py                          1-hour-window LSTM (INPUT_STEPS=4) -> models_1h/, results_1h/
    train_arima.py, compile_results.py        ARIMA baseline + comparison table
    extract_real_iawe.py, generate_synthetic_iawe.py   dataset prep (iAWE, Delhi 2013)
    run_control_simulation.py                 offline control simulation
    control_logic.py                          decide(): thresholds, Tariff A brackets, ToU peak tightening
    tou_pricing.py                            TNB ToU periods/prices, cost helpers
    live_control_loop.py                      LIVE loop (15-min), Blynk I/O, scaling, ToU push. BLYNK_TOKEN goes here
    live_control_loop_1h.py                   same with 1-hour models
    live_control_fast.py                      fast test loop (reads V0 every 30 s), logs data/live_fast/
    demo_decision_now.py                      instant scenario decisions, --push to Blynk, ToU scenarios
    day_view.py                               data for dashboard "Today" tab
    dashboard.py                              Streamlit dashboard (streamlit run dashboard.py)
  models/, models_1h/        lstm_fan.keras, lstm_geyser.keras, lstm_computer.keras
  results/, results_1h/      metrics JSON, plots, comparison_table.csv
  data/raw/                  iAWE channel CSVs (fan=TV channel, geyser=AC channel, computer=laptop)
  data/processed(_1h)/<app>/ resampled_15min.csv, scaler.pkl, X/y npy splits
  data/live*/                created at runtime (buffers, decisions.csv, readings.csv)
  enclosure/                 3D-printed controller box: generator .py, base/lid STL, layout PNG
Zip of everything: ..\ilyasa_lstm_update_2026-10-07.zip
```

## 5. Key design facts
- LSTM: 2x LSTM(64), dropout 0.2, Adam/MSE, 15-min buckets, 2-step (30 min) output, decision uses max of the 2 steps.
  1h models: geyser RMSE 427.9 W R2 0.696; fan R2 0.302; computer RMSE 12.6 W R2 0.505 (test sets differ per model).
- Appliance mapping: fan model = iAWE TV channel; water heater = AC channel (proxy); lamp uses the fan model.
- Wattage mapping (V15): inputs x k, forecast / k, threshold / k, k = training typical / real typical.
  Fan 50 W -> 74 W (x1.48, limit ~57 W). Lamp 3 W -> 74 W (x24.67, limit ~3.45 W). Heater not scaled (limit 1620 W).
- Policies: heater + lamp = hard CUTOFF (30 s buzzer warning first); fan = SHIFT (advisory: beeps + LCD + reason,
  relay stays on); laptop = NOTIFY only.
- Tariff: Tariff A mode tightens thresholds near monthly brackets 200/300/600/900 kWh (old rates, report only).
  ToU mode: peak weekdays 2pm-10pm, all-in 45.92 sen/kWh; off-peak 41.83; limits x0.8 at peak.
  Normal tariff since July 2025: 44.43 sen/kWh all-in. Whether Ilyasa's house is on ToU is UNKNOWN.
- Firmware fail-safe: cloud lost > 5 min or no Python command > 45 min -> relay ON, manual/buzzer reset.
- Message budget: account limit 100k/month. LIVE_SEND_MS 5 min normally; 5000UL only for testing.

## 6. How to run
1. pip install -r requirements.txt (Python 3.10+). Set BLYNK_TOKEN in scripts/live_control_loop.py.
2. Arduino IDE libraries: Blynk (Volodymyr Shymanskyy), PZEM004Tv30 (Jakub Mandula), LiquidCrystal I2C
   (Frank de Brabander), DHT sensor library (Adafruit) + Adafruit Unified Sensor; board package esp32 (Espressif).
   In the sketch set BLYNK_TEMPLATE_ID "TMPL6a5gyryUK", auth token, WiFi SSID/pass. Flash.
3. In scripts/: python live_control_fast.py --appliance lamp   |   streamlit run dashboard.py
4. Instant demo: python demo_decision_now.py --all  /  --tou-scenario fan_peak --push

## 7. Status and open items
Done: hardware reads OK (243 V, 3.3 W lamp), buzzer OK, Blynk template/dashboard, live loops, 1h models,
scaling, manual controls, DHT22, ToU pricing, dashboard Today tab (power + price, actual/historical/LSTM), Blynk history fix.
Open:
- Relay switching fix not confirmed (section 2).
- Last seen: Arduino "BlynkSimpleEsp32.h: No such file" -> libraries not installed on client PC; user also had
  TEMPLATE_ID and token still as placeholders in a new unsaved sketch.
- SHIFT is advisory only; offered timed pause + Blynk push notification (not built).
- Anomaly detection (agreed "feature 3") not built yet.
- Confirm Ilyasa's actual TNB tariff; update report Ch2.2 tariff figures; Ch4 results from live runs.

## 8. Working notes for Claude
- Use simple wording with Aizziq; give exact pins/steps. Mains safety: unplugged work, lamp first.
- Never paste or store the Blynk auth token.
- Be honest in report framing: heater model is an AC-proxy; live demo = mechanism demonstration;
  wattage mapping fixes amplitude only, not usage pattern.
