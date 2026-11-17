# Client walkthrough video: script and shot list

Ilyasa PSM2, Smart Home Energy Forecasting with LSTM. About 11 minutes, 21 scenes.

How to use this:
- **Ready-made video:** `walkthrough.mp4` shows every slide below with the narration as on-screen captions (no voice). `walkthrough_captions.srt` has the same text as a subtitle file.
- **Add your voice:** play the MP4, read the narration aloud, and record it (DaVinci Resolve or phone). The caption timing follows normal speaking speed.
- **Record your own screen instead:** follow the *Show on screen* line for each scene and read the narration.
- **Never show the Blynk auth token** (Device Info page, `BLYNK_TOKEN` line in `live_control_loop.py`, token line in the sketch). Blur or crop it.

## Scene 1: Smart Home Energy Forecasting with LSTM

**Show on screen:** Title slide, or the project folder open in File Explorer.

**Narration:** Hi Ilyasa. This video walks you through your PSM2 project. We will cover four things. What the system does. How it is wired. Which commands to run and how to read the console. And how to read the machine learning results.

## Scene 2: What the system does

**Show on screen:** The flow diagram. Point at each box as you say it.

**Narration:** The PZEM sensor measures how much power one appliance is using. The ESP32 sends that reading over WiFi to Blynk. A Python script on the laptop reads it from Blynk, and an LSTM model predicts the power for the next thirty minutes. The script compares the prediction with a limit, makes a decision, and writes it back to Blynk. The ESP32 picks up the decision and acts, using the relay, the buzzer and the LCD. The laptop never talks to the ESP32 directly. Everything goes through Blynk.

## Scene 3: The four decisions

**Show on screen:** The decision table.

**Narration:** There are four possible decisions. OK means no action. NOTIFY is a warning only. It is used for the laptop, which is never cut. SHIFT is advice to use the appliance at a cheaper time. It is used for the fan, and the relay stays on. CUTOFF is for the water heater and the test lamp. The buzzer warns for thirty seconds, then the relay cuts the power. The limit gets tighter when this month's usage is close to the next TNB price bracket, or during the Time of Use peak, from 2 pm to 10 pm on weekdays.

## Scene 4: Wiring: the whole circuit

**Show on screen:** firmware/circuit_image.png full screen.

**Narration:** Here is the full circuit. On the left is the low voltage side: the ESP32, the LCD, the DHT22 sensor and the buzzer. On the right is the mains side: the PZEM with its current coil, the relay, and the socket where the appliance plugs in. The lamp in this picture stands for whatever appliance is plugged into the socket.

## Scene 5: Wiring: ESP32 pins (low voltage)

**Show on screen:** The pin table, or firmware/Wiring_Diagram.svg zoomed on the controller box.

**Narration:** These are the pins. GPIO 26 goes to the relay input. GPIO 27 goes to the buzzer. The PZEM uses serial: its TX goes to GPIO 16, and its RX goes to GPIO 17. The LCD uses I2C on GPIO 21 for SDA and GPIO 22 for SCL. The DHT22 data pin goes to GPIO 4. The relay, the PZEM and the LCD take power from VIN, which is 5 volts. And every ground is joined together.

## Scene 6: Wiring: mains side (socket box)

**Show on screen:** firmware/Wiring_Diagram.svg, top half (the red dashed mains box).

**Narration:** On the mains side, only the live wire is switched. From the plug, live goes through the 10 amp fuse and through the CT coil, into the relay COM terminal. The relay NO terminal goes to the socket live. Neutral and earth go straight through. The PZEM voltage terminals tap live and neutral before the relay, so voltage is always measured.

## Scene 7: Safety and one open check

**Show on screen:** This checklist. If you have the hardware, show the lamp test.

**Narration:** Safety first. Always unplug the box before you open or touch it. Test with the lamp first, then the fan, and only then the water heater. One thing to check before your demo: the relay board is a 5 volt active low board, and with a 3.3 volt signal it can stay on all the time. Fix it with the JD VCC jumper, or set RELAY OPEN DRAIN to true in the sketch. Then run the pin test sketch in mode 2, and make sure the relay clicks on and off.

## Scene 8: Blynk: the pins to know

**Show on screen:** Blynk web dashboard on Devices > Ilyasa Energy. Do NOT open Device Info on camera (the token is there).

**Narration:** In Blynk, V0 is the live power. V10 is the decision, from 0 for OK up to 3 for CUTOFF, and V11 is the reason text. V12 shows the relay state. V13 lets you force the relay on or off by hand, and V14 lets you mute the buzzer. V15 turns the wattage mapping on or off, and V16 to V20 show the tariff and the cost. The switches only work on the Devices page, not on the template page. Your auth token is in Device Info. Keep it private, and never show it in a recording.

## Scene 9: Commands: one-time setup

**Show on screen:** Command Prompt in the scripts folder; Arduino IDE with the main sketch (blur the token line).

**Narration:** Setup is done once. Open a Command Prompt in the scripts folder and install the Python packages with pip. Then open live control loop dot py and paste your Blynk token. In the Arduino IDE, open the main sketch, choose ESP32 Dev Module, fill in the template ID, the token and your WiFi, and upload. For quick testing, set LIVE SEND MS to five thousand, so the ESP32 sends a reading every five seconds.

## Scene 10: Commands: running it

**Show on screen:** Two Command Prompt windows side by side, then the dashboard in the browser.

**Narration:** To run it, use two terminals. In the first one, start the live loop and tell it which appliance is plugged in, for example the lamp. In the second, start the dashboard with streamlit, or just double click run dashboard dot bat. It opens in your browser. If you have no hardware with you, demo decision now gives instant results. Add dash dash push to send the result to Blynk, and the LCD will show it.

## Scene 11: Reading the console: the header

**Show on screen:** Terminal 1 right after starting live_control_fast.py.

**Narration:** When the live loop starts, it prints a header. The first line says if wattage mapping is on. The small lamp is scaled up to the size of appliance the model was trained on. The second line shows the tariff and the price right now. Fast test mode means it reads every thirty seconds. That tests the whole chain, but it is not a true thirty minute forecast. The last line shows the appliance and this month's kilowatt hours, which are used for the TNB bracket check.

## Scene 12: Reading the console: each reading

**Show on screen:** Terminal 1 after about two minutes.

**Narration:** The first four readings only fill the window, so there is no decision yet. From the fourth reading, each line shows three numbers. Now is the live reading. Forecast is what the LSTM expects next. Limit is the threshold. If the forecast is above the limit, the decision changes. The decision is sent to Blynk only when it changes, so the buzzer does not beep every thirty seconds. If you see could not read V0, check that the ESP32 is on, the WiFi, and the token. If V0 has not changed for six readings, set LIVE SEND MS to five thousand in the sketch.

## Scene 13: Reading the LCD

**Show on screen:** The real LCD (phone camera), or these mock screens.

**Narration:** The LCD has two rows. The top row shows the decision. The star at the end means Blynk is connected. An exclamation mark means it is not. The second row shows power, voltage and temperature, or the reason text when there is a warning. CUTOFF in 25 seconds is the countdown before the relay cuts. FAILSAFE means the cloud was lost for more than five minutes, or Python sent nothing for 45 minutes, so the relay is switched back on to be safe.

## Scene 14: The ML model in one slide

**Show on screen:** This slide, or scripts/train_lstm_1h.py open in an editor.

**Narration:** Now the machine learning. The models were trained on the iAWE dataset, which is real power data from homes in Delhi, in fifteen minute steps. The model looks at the last four readings, which is one hour, and predicts the next two steps, which is thirty minutes. It is two LSTM layers of 64 units with dropout. There is one model per appliance. Be honest about this part: the water heater model was trained on an air conditioner channel, so it is a proxy, and the fan model was trained on a TV channel. ARIMA is used as the baseline to compare against.

## Scene 15: Prediction vs actual (water heater, 1-hour model)

**Show on screen:** results_1h/geyser_prediction_vs_actual.png

**Narration:** This graph is the easiest way to see if the model works. Blue is the real power, and orange is the prediction, on test data the model never saw. When the heater is steadily on or off, the orange line sits right on the blue line. When the heater suddenly switches, the prediction is about one step late. That is normal for this kind of model, because it can only learn from what it has already seen.

## Scene 16: Training loss curve

**Show on screen:** results_1h/geyser_loss_curve.png

**Narration:** The loss curve shows the training. Blue is the error on training data, and orange is the error on validation data the model did not train on. Both drop quickly and stay low. Because the orange line does not climb back up, the model is not overfitting.

## Scene 17: How to read the numbers (1-hour models)

**Show on screen:** Dashboard Model tab, or results_1h/*_lstm_results.json

**Narration:** Here are the numbers. RMSE and MAE are the average error in watts, so smaller is better. Compare them with the size of the appliance. An error of four hundred watts sounds big, but the heater swings between zero and almost two thousand watts. R squared is a score where one is perfect, and zero means no better than guessing the average. The water heater model is the strongest, at 0.70. The laptop is 0.50. The fan is the weakest at 0.30, because its power hardly changes, so there is very little pattern to learn.

## Scene 18: LSTM vs ARIMA (be honest)

**Show on screen:** results/lstm_vs_arima_rmse.png and results/comparison_table.csv

**Narration:** This chart compares the LSTM with ARIMA. Be careful with it. The two models were tested on different samples, so the RMSE bars are not a fair head to head. Using R squared, the LSTM is better for the water heater, 0.64 against 0.43. ARIMA is better for the laptop. Say this openly in your report and viva. It is a real result, not a mistake.

## Scene 19: Seeing the decision change

**Show on screen:** Dashboard: Control panel > scenario heater_normal, then heater_near_tariff. Then the Decision logic tab.

**Narration:** This is the best thing to show in your viva. It is the same water heater, using the same fifteen hundred watts. In the middle of the month, the limit is sixteen hundred and twenty watts, so the system only gives a warning. When the month's usage is just five kilowatt hours from the two hundred bracket, the limit drops to twelve hundred and fifty five watts, and the same heater is cut off. That is the predictive, tariff aware part working.

## Scene 20: The dashboard tabs

**Show on screen:** Open each dashboard tab in turn (run_dashboard.bat).

**Narration:** Finally, the dashboard. The control panel is where you start a demo or the live loop. The Today tab shows power and price for today, with the real reading in blue and the LSTM in orange. Live decisions is the main demo page. Model shows the accuracy, and Decision logic lets you try your own numbers. When you finish, press stop live loop before you close the dashboard.

## Scene 21: Say it honestly

**Show on screen:** This slide.

**Narration:** To finish, a few things to say honestly to your panel. The heater model is an air conditioner proxy. The live demo shows the mechanism working from end to end. It is not a measured saving. Wattage mapping fixes the size of the reading, not the usage pattern. And demo scenarios are not live sensor data. That is the whole project. Good luck with your viva.
