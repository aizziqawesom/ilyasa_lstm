# Stupid simple run guide (Ilyasa PSM2, LSTM energy control)

Read this top to bottom once. After that, the "Cheat sheet" at the end is all you need.

---

## 0. What is this thing, in one breath

A smart socket measures how many watts an appliance is using. Every few minutes that number goes to the internet (Blynk). A Python program on the laptop grabs the number, asks the **LSTM** "what will this appliance use in the next 30 minutes?", compares the answer with a limit, and decides **OK / NOTIFY / SHIFT / CUTOFF**. It sends the decision back through Blynk, and the ESP32 beeps, shows it on the LCD, and (for CUTOFF) switches the relay off.

```
Appliance -> PZEM (measures W) -> ESP32 -> WiFi -> Blynk cloud
                                                     |
                                   laptop: Python + LSTM decides
                                                     |
ESP32 <- WiFi <- Blynk cloud <-----------------------+
  -> relay / buzzer / LCD
```

The laptop never talks to the ESP32 directly. Blynk is the middleman both ways.

## 1. What the LSTM actually does (plain words)

- It is a small neural network that is good with sequences (numbers over time).
- **Input:** the last 1 hour of power readings, as 4 numbers (one per 15 minutes). The older "2-hour" model uses 8 numbers.
- **Output:** 2 numbers, its guess of the power for the next 15 min and the 15 min after that (so 30 min ahead). The system uses the bigger of the two.
- **It does not decide anything by itself.** It only forecasts. The decision is plain rules in `scripts/control_logic.py`: if the forecast is above the limit, act.
- **The limit gets stricter** when the month's usage is close to the next TNB price step (200 / 300 / 600 / 900 kWh), and also during ToU peak hours (weekdays 2pm to 10pm, limit x0.8).
- **What it learned from:** the iAWE dataset (a real house in Delhi, 2013). There was no fan or water heater channel, so:
  - fan model = trained on the iAWE **TV** channel
  - water heater model = trained on the iAWE **air-con** channel (it is a proxy, say so honestly)
  - lamp = uses the fan model
  - laptop = iAWE laptop channel (exact match, but no live hardware for it)
- **Wattage mapping:** a 3 W lamp or 50 W fan is much smaller than what the model saw in training. So the readings are scaled up before the model, and the answer is scaled back down. This fixes the size only, not the usage pattern.

What each decision means:

| Decision | Colour | What happens | Used for |
|---|---|---|---|
| OK | green | nothing | all |
| NOTIFY | yellow | warning only | laptop (never cut) |
| SHIFT | orange | beeps + LCD "use it later". Relay stays ON (advice only) | fan |
| CUTOFF | red | buzzer for 30 s, then relay switches OFF | water heater, test lamp |

**Good news: the models are already trained.** They are in `models/` (2-hour) and `models_1h/` (1-hour). You do NOT need to train anything to run the demo. Training is in section 7, only if you want to redo it.

---

## 2. Install (once only)

1. Install **Python 3.10 or newer** (tick "Add Python to PATH" in the installer).
2. Open Command Prompt in the `ilyasa_lstm\scripts` folder (in File Explorer, click the address bar, type `cmd`, press Enter).
3. Type:

   ```
   pip install -r ..\requirements.txt
   ```

   This installs pandas, numpy, scikit-learn, tensorflow, statsmodels, matplotlib, requests, streamlit. Takes a few minutes.

4. Put the Blynk token in the Python side. Open `scripts\live_control_loop.py` and find this line (around line 96):

   ```python
   BLYNK_TOKEN = "YOUR_BLYNK_AUTH_TOKEN"
   ```

   Replace `YOUR_BLYNK_AUTH_TOKEN` with the token from **Blynk > Devices > Ilyasa Energy > Device Info**. Save.
   Do not send this token to anyone or paste it in chats or the report.

That is all for the laptop. **If you only want the no-hardware demo, skip to section 4.**

---

## 3. Flash the ESP32 (once, or after changing the sketch)

### 3a. Arduino IDE setup (once)
1. Install Arduino IDE.
2. Boards Manager: install **esp32 by Espressif**. Board = **ESP32 Dev Module**.
3. Library Manager: install these 5:
   - **Blynk** (Volodymyr Shymanskyy)
   - **PZEM004Tv30** (Jakub Mandula)
   - **LiquidCrystal I2C** (Frank de Brabander)
   - **DHT sensor library** (Adafruit)
   - **Adafruit Unified Sensor**

   If you see `BlynkSimpleEsp32.h: No such file`, it means the Blynk library is missing. Install it.

### 3b. Fill in the sketch
Open `firmware\ilyasa_esp32_main\ilyasa_esp32_main.ino` (the folder and file name must match, Arduino needs that).
Do NOT use `firmware\relay_control.ino`, it is old.

Change these lines near the top:

```cpp
#define BLYNK_TEMPLATE_ID   "TMPL6a5gyryUK"          // was "TMPLxxxxxxxxx"
#define BLYNK_AUTH_TOKEN    "YOUR_BLYNK_AUTH_TOKEN"  // put the same token as in live_control_loop.py
const char WIFI_SSID[] = "YOUR_WIFI_SSID";           // 2.4 GHz WiFi only
const char WIFI_PASS[] = "YOUR_WIFI_PASSWORD";
```

For a quick live test, also change:

```cpp
const unsigned long LIVE_SEND_MS = 5000UL;   // was 5UL * 60UL * 1000UL
```

(5000 = send every 5 s. Good for testing, but it uses many Blynk messages. Put it back to 5 minutes for long runs; the account limit is 100k messages a month.)

Plug the ESP32 in by USB, pick the COM port, press **Upload**. Open Serial Monitor at **115200** baud. You should see `WiFi=connected  Blynk=connected` and PZEM readings.

### 3c. Pins (already wired, for checking only)

| Part | ESP32 pin |
|---|---|
| Relay IN | GPIO26 |
| Buzzer + | GPIO27 |
| PZEM TX -> | GPIO16 (RX2) |
| PZEM RX <- | GPIO17 (TX2) |
| LCD SDA / SCL | GPIO21 / GPIO22 (address 0x27, try 0x3F if blank) |
| DHT22 data | GPIO4 (3.3 V) |
| Relay, PZEM, LCD VCC | VIN / 5V |
| All GND | common GND |

### 3d. Safety (do not skip)
- Do all wiring with the plug **pulled out of the wall**.
- Test with the **3 W lamp first**. Only use the fan or heater after the lamp works properly, including the relay cut.
- Use an RCCB-protected wall socket. Full mains wiring is in `firmware\BUILD_GUIDE.md`.

### 3e. Known relay problem (not confirmed fixed)
The relay board is 5 V "active-low". From a 3.3 V ESP32 pin it can stay ON all the time. Two fixes, pick one:
- **(a) Jumper fix:** remove the JD-VCC jumper, JD-VCC to 5 V, VCC to 3.3 V, then in the sketch set `RELAY_ACTIVE_HIGH = false`.
- **(b) Software fix:** in the sketch set `RELAY_ACTIVE_HIGH = false` and `RELAY_OPEN_DRAIN = true`.

Test it with `firmware\pin_test\pin_test.ino` (MODE 2 = open-drain) before trusting CUTOFF.

---

## 4. Open the dashboard (every time)

**Windows:** double-click **`run_dashboard.bat`** in the `ilyasa_lstm` folder.
**Mac:** the .bat file does not work on a Mac. Use `run_dashboard.command` instead, see section 4c.

A black window opens and the browser opens **http://localhost:8501** by itself (if not, type that address). First load is slow because TensorFlow loads the model. To close: stop the live loop first (if running), then close the black window.

(Same thing by hand: in `scripts`, run `python -m streamlit run dashboard.py`.)

Now pick ONE of these:

### 4a. Demo without hardware (works anywhere, viva-safe)
1. Tab **Control panel** > choose **Demo without hardware**.
2. Press **Fill dashboard with simulated data**. Wait for it to finish.
3. Open the **Live decisions** tab. You now see lines and coloured decisions.
4. Optional, back in Control panel > **Run scenario**. Good ones to show:
   - `heater_normal` (1500 W, 150 kWh this month) -> NOTIFY
   - `heater_near_tariff` (same heater, 195 kWh) -> CUTOFF, because the limit got stricter near 200 kWh
   - `fan_offpeak` (11 am) -> OK
   - `fan_peak` (3 pm weekday) -> SHIFT
5. **Clear simulated / live logs** wipes it again.
6. Optional, **Send demo to Blynk** (Control panel, block 3): pushes the same kind of demo to Blynk so the Blynk app shows it too (power, V21/V22 chart lines, decision, reason, cost). **Unplug the ESP32 first**, otherwise it overwrites V0 with the real reading and acts on the demo decisions (buzzer, relay). Takes about 2 minutes (6 simulated hours, one 15-min step every 5 s), about 190 Blynk messages, and sends OK at the end. Same from the command line: `python demo_to_blynk.py` (add `--appliance fan`, `--tariff tou`).

Say clearly when presenting: this is a **scenario run through the real LSTM and real rules**, not live sensor data.

### 4b. Live with the real hardware
Needs: ESP32 flashed and powered, showing `Blynk=connected`, token set in `live_control_loop.py`, lamp plugged into the socket box.

1. Tab **Control panel** > choose **Live with hardware (ESP32 + Blynk)**.
2. Set:
   - **Appliance plugged in:** lamp / fan / geyser (what is really plugged in)
   - **Seconds between readings:** `30` for a quick test, `900` for the real 15-minute mode
   - **Month's kWh so far:** e.g. 150
   - **Tariff:** TNB ToU or Normal (A)
3. Press **Start live loop**. First decision arrives in about 2 minutes (at 30 s). At 900 s it takes about 1 hour.
4. Watch the **Live decisions** tab, the LCD, and the Blynk app.
5. Press **Stop live loop** before closing the dashboard.

Important honesty note: at 30 s steps the model is fed 30-second readings although it was trained on 15-minute steps, so it is **not a true 30-min forecast**. It proves the whole chain works end to end. For Chapter 4 results, use the real 15-minute loop (section 6).

---

### 4b-demo. Real ESP32, fake readings (no mains needed)
Sketch: `firmware\ilyasa_esp32_demo\ilyasa_esp32_demo.ino`. Same as the main sketch, but the ESP32 makes up the PZEM readings itself (water heater 1500 W by default, with a short 15 % spike every 5 minutes so the LSTM has something to react to). Everything after the reading is real: Blynk, Python LSTM, decision, buzzer, LCD, relay. If the relay cuts, the fake power drops to 0 W, so the cut shows on the Blynk chart.
1. Fill in the token and WiFi (template ID is already set), pick `DEMO_APPLIANCE` (0 heater, 1 fan, 2 lamp), flash it. USB power only; leave the relay output disconnected from mains.
2. LCD shows `Ilyasa DEMO MODE` at boot and `DEMO 1512.3W` on row 2.
3. Laptop: Control panel > Live with hardware > appliance matching `DEMO_APPLIANCE` (geyser / fan / lamp) > 30 s > Start. Or `python live_control_fast.py --appliance geyser --kwh 195` to see CUTOFF quickly.
4. Sends V0 every 5 s (about 720 Blynk messages an hour), so unplug it when done. Flash `ilyasa_esp32_main` again for real readings.
Say it plainly when showing it: the readings are simulated by the ESP32; the rest of the chain is real.

### 4c. Running on a Mac

Use Python **3.11 or 3.12** from python.org (TensorFlow may not install on the newest Python). Check with `python3 --version`.

**Easy way (double-click):**
1. Put `run_dashboard.command` in the `ilyasa_lstm` folder (same place as `run_dashboard.bat`).
2. Open **Terminal** once and make it runnable (drag the `ilyasa_lstm` folder into Terminal after typing `cd `):
   ```
   cd /path/to/ilyasa_lstm
   chmod +x run_dashboard.command
   ```
3. Double-click `run_dashboard.command`. If macOS says it "cannot be opened", right-click it > **Open** > **Open** (only the first time).
4. The first run creates a `.venv` folder and installs everything (a few minutes). After that it opens http://localhost:8501 straight away.
5. To stop: Ctrl+C in that Terminal window, or close it.

**By hand in Terminal (same result):**
```
cd /path/to/ilyasa_lstm
python3 -m venv .venv                 # once
source .venv/bin/activate             # every new Terminal window
pip install -r requirements.txt       # once
cd scripts
python -m streamlit run dashboard.py  # dashboard
```

Live loop / other scripts on Mac: open a second Terminal, then
```
cd /path/to/ilyasa_lstm
source .venv/bin/activate
cd scripts
python live_control_fast.py --appliance lamp
```
Every command in section 6 works the same on Mac, just run it after `source .venv/bin/activate`. Use `/` instead of `\` in paths.

Arduino IDE, the ESP32 sketch and Blynk are the same on Mac. The ESP32 port shows as `/dev/cu.usbserial-...` or `/dev/cu.SLAB_USBtoUART`; if no port appears, install the CP210x or CH340 USB driver for your board.

---

## 5. "What the hell are those dashboards?"

There are **two different dashboards**. Don't mix them up.

### Dashboard 1: Streamlit (on the laptop, in the browser, localhost:8501)
This is the **ML / explanation dashboard**. It mostly reads and shows. Only the Control panel starts things. It is for the viva, for screenshots in the report, and for checking the model.

**Left sidebar**
- **Model set:** 1-hour (use this) or 2-hour. Both forecast 30 min ahead.
- **Appliance:** which appliance the Live / Dataset / Model / Decision logic tabs show.
- **Live tab auto-refresh:** how often the page reloads the log files.
- **Wattage mapping:** on/off for the what-if tester only (explained in section 1).
- Below it: the scale factor and the limit (threshold) for the chosen appliance.

**Tabs, one by one**

| Tab | What it shows | What it is for |
|---|---|---|
| **Control panel** | Buttons: simulated demo, instant scenarios, start/stop live loop | Running things without typing commands |
| **Today: power & price** | For each appliance, 2 charts: power today and cost today | Showing "what today looks like vs a normal day vs what the LSTM expects", and money |
| **Live decisions** | Every decision the system made, with the numbers | **Main demo page.** Proves the forecast > limit > action chain |
| **Dataset** | The iAWE training data, pick any day | Explaining where the model learned from (not live) |
| **Model** | LSTM vs ARIMA accuracy table, replay chart, loss curves | Proving how good (or not) the model is. Chapter 4 material |
| **Decision logic** | Chart of how the limit drops as monthly kWh rises, plus a what-if box | Explaining the rules; type your own 4 readings and see the decision |

**How to read the lines**

Today tab:
- Blue solid = actual readings today
- Grey dashed = the "average normal day" from training data
- Orange dashed = LSTM forecast
- Red vertical line = now
- Grey band = ToU peak (2pm to 10pm weekdays)
- Purple line = electricity price (sen/kWh)
- Only the appliance you are actually running has a blue line. The others show grey + orange only.

Live decisions tab:
- Top boxes: latest reading, LSTM forecast, limit now, decision
- Blue solid = actual power
- Orange dashed = LSTM forecast for the next 30 min, made at that moment
- Grey dotted = the limit (goes down near a TNB bracket)
- Coloured ticks under the chart = decision at that time (green / yellow / orange / red)
- Tables: decision counts, and recent decisions with the reason text
- Temperature / humidity chart from the DHT22 (if logged)
- "Fetch current values from Blynk" = reads what Blynk has right now (read only)

Model tab:
- **RMSE / MAE** = average error in watts. Smaller is better.
- **R2** = 1 is perfect, 0 is no better than guessing the average, negative is worse.
- Honest result: **LSTM wins on the water-heater proxy; ARIMA wins on the laptop and the fan's RMSE.** Say that as it is.
- 1-hour model numbers: heater RMSE about 428 W (R2 0.70), fan R2 0.30, laptop RMSE about 12.6 W (R2 0.51).

### Dashboard 2: Blynk (phone app or blynk.cloud website)
This is the **device / remote control dashboard**. It shows live sensor values and lets you control the socket by hand. Open it at **Devices > Ilyasa Energy** (controls do not work on the template page).

| Widget | Pin | Meaning |
|---|---|---|
| Power | V0 | watts right now |
| Energy | V3 | kWh measured |
| 15-min average | V5 | average W (this is what the 15-min loop uses) |
| Temperature / humidity | V6 / V7 | DHT22 |
| Decision | V10 | 0 OK, 1 NOTIFY, 2 SHIFT, 3 CUTOFF |
| Reason | V11 | text explaining the decision |
| Relay state | V12 | on / off |
| Manual relay | V13 | 0 AUTO, 1 force ON, 2 force OFF |
| Buzzer | V14 | 0 AUTO, 1 ON, 2 MUTE |
| Wattage mapping | V15 | 0 off, 1 on |
| ToU period / price / cost per hour | V16 / V17 / V18 | tariff info |
| ToU switch | V19 | 1 ToU, 0 normal tariff |
| Cost today | V20 | RM today |
| Power history chart | V0 | power over time |
| Pricing history chart | V17 + V18 | price and cost over time |
| LSTM expected (chart line) | V21 | what the LSTM predicted, one step earlier, the power would be now |
| Historical avg (chart line) | V22 | iAWE average for this time of day, in the real appliance's watts |
| Limit (chart line) | V23 | the threshold the decision used, in real watts (drops near a bracket / at ToU peak) |

**Adding the LSTM, historical and limit lines to the Power history chart (one time):**
1. Blynk website > **Developer Zone > My Templates > Ilyasa Energy > Datastreams > + New Datastream > Virtual Pin**:
   - `V21`, name **LSTM expected**, Data type **Double**, Units **Watt**, Min 0, Max 2000, Decimals #.##
   - `V22`, name **Historical avg**, Data type **Double**, Units **Watt**, Min 0, Max 2000, Decimals #.##
   - `V23`, name **Limit**, Data type **Double**, Units **Watt**, Min 0, Max 2000, Decimals #.##
   - Use the same Max as V0. For both, set history/"Save raw data" the same as V0 (**1 Minute AVG**).
2. **Web Dashboard** tab (and the phone app layout if you use it): edit the **Power history** chart > **Add datastream** > V21, then V22, then V23. Make V0 blue solid, V21 orange (dashed if offered), V22 grey, V23 red. Save and **Apply** the template.
3. Nothing to change on the ESP32. The Python loops (`live_control_fast.py`, `live_control_loop_1h.py`, Control panel live loop) send V21, V22 and V23 every cycle. If the datastreams don't exist yet they print one note and carry on.

How to read it: V21 is sent one step late on purpose, so the orange line sits right on top of the blue line it was predicting. Where orange and blue are close, the LSTM predicted well. Grey is "a normal day at this time" from the training data (an average of on and off periods, so it can sit below a running heater). The decision itself still uses the 30-min-ahead forecast, which is in the Streamlit Live tab.

Leave Manual relay on **AUTO** if you want the LSTM to be in control.

### The LCD on the box
- Top line: the decision, e.g. `Status: OK` or `SHIFT  PEAK`, or `CUTOFF in 25s` during the warning, or `MANUAL: ON/OFF`, or `FAILSAFE: ON`.
- Last character: `*` = connected to Blynk, `!` = not connected.
- **Failsafe:** if the ESP32 loses Blynk for more than 5 minutes, or gets no decision from Python for 45 minutes, it turns the relay ON (appliance back on) so nothing stays cut by accident.

---

## 6. Running from the command line instead (optional)

All commands are typed inside `ilyasa_lstm\scripts`.

| Want to | Command |
|---|---|
| Quick live test (30 s steps) | `python live_control_fast.py --appliance lamp` |
| Same, near a tariff bracket | `python live_control_fast.py --appliance lamp --kwh 195` |
| **Real live loop for Ch4 (1-hour model, every 15 min)** | `python live_control_loop_1h.py --appliance lamp` |
| Real live loop, one cycle only | `python live_control_loop_1h.py --once` |
| Old 2-hour model loop | `python live_control_loop.py --appliance fan` |
| Instant scenario, no waiting | `python demo_decision_now.py --scenario heater_near_tariff` |
| Same, and send it to the ESP32 | `python demo_decision_now.py --tou-scenario fan_peak --push` (LCD shows `SHIFT  PEAK`) |
| List all scenarios | `python demo_decision_now.py --list` |
| Run all scenarios | `python demo_decision_now.py --all` |
| Fill dashboard with fake data | `python simulate_live.py` |
| Compare fixed threshold vs LSTM (offline) | `python compare_fixed_vs_lstm.py` (results in `results_1h\fixed_vs_lstm.csv`) |
| Use normal tariff instead of ToU | add `--tariff a` |
| Turn wattage mapping off | add `--no-scaling` |

Stop any loop with **Ctrl+C**.

Warm-up times: the 1-hour loop needs 4 readings 15 minutes apart, so the first decision comes after about 45 to 60 minutes. The 2-hour loop needs about 2 hours.

Where results are saved:
- `data\live_1h\decisions.csv`: every decision from the real 1-hour loop (use this for Ch4)
- `data\live_fast\fast_test_log.csv` and `readings.csv`: fast test loop
- `data\live\decisions.csv`: 2-hour loop

---

## 7. Training the LSTM again (only if you want to, not needed for the demo)

In `ilyasa_lstm\scripts`, in this order:

```
python preprocess.py --all          # cleans iAWE data, makes 15-min windows (2-hour model)
python train_lstm.py --all          # trains 2-hour models -> models\
python train_lstm_1h.py --all       # trains 1-hour models -> models_1h\  (the one the demo uses)
python train_arima.py --all         # ARIMA baseline for comparison
python compile_results.py           # makes results\comparison_table.csv and the RMSE chart
python run_control_simulation.py --all   # offline control simulation on test data
```

Each one prints its scores. Training takes a few minutes per appliance on a normal laptop. Retraining will give slightly different numbers each time, so if you retrain, update the numbers in the report.

---

## 8. If something goes wrong

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Redo section 2 step 3 (`pip install`) |
| `Set BLYNK_TOKEN in live_control_loop.py first` | Section 2 step 4 |
| `BlynkSimpleEsp32.h: No such file` | Install the Blynk library (section 3a) |
| Serial Monitor: WiFi NOT connected | Check SSID / password; must be 2.4 GHz |
| LCD blank | Change `LCD_ADDR` to `0x3F`, check SDA/SCL, turn the contrast screw on the LCD back |
| LCD shows `!` | ESP32 not connected to Blynk (WiFi or token) |
| Live decisions tab empty | Nothing has run yet: use the simulated demo or start the live loop and wait |
| Relay never switches off | The active-low problem, section 3e |
| Blynk charts empty | History must be set to "1 Minute AVG" (already done); old data can't be recovered |
| Live loop still running after closing the browser | Press Stop first next time, or end the Python process in Task Manager |

---

## 9. Known open issues (be honest about these)
- **Relay active-low fix is not confirmed yet** (section 3e). Test before showing CUTOFF.
- **SHIFT is advice only.** The fan keeps running; the system just beeps and shows a message.
- The water heater model is trained on an **air-con** channel (proxy). The fan model is trained on a **TV** channel.
- The live demo shows the **mechanism works**. It is not a months-long savings study.
- It is not known yet whether Ilyasa's house is on TNB ToU. ToU savings only count if it is.

---

## Cheat sheet (viva in 2 minutes)

1. Double-click `run_dashboard.bat`.
2. Control panel > Demo without hardware > **Fill dashboard with simulated data**.
3. **Live decisions** tab: point at blue (actual), orange (forecast), grey (limit), coloured ticks (decision).
4. Control panel > scenario `heater_normal` (NOTIFY), then `heater_near_tariff` (CUTOFF): "same heater, but close to 200 kWh the limit tightens".
5. **Decision logic** tab: show the limit dropping near 200 kWh.
6. **Model** tab: LSTM vs ARIMA, said honestly.
7. With hardware: Live with hardware > lamp > 30 s > Start. Show the LCD and Blynk app reacting.
