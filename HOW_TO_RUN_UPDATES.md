# How to run the latest updates (ToU pricing + Today dashboard)

## 1. One-time setup
1. Unzip, open a terminal in `ilyasa_lstm/scripts`.
2. `pip install -r ../requirements.txt` (needs Python 3.10+).
3. In `live_control_loop.py` set `BLYNK_TOKEN = "..."` (Blynk > Devices > Ilyasa Energy > Device Info, copy the token).
4. Flash `firmware/ilyasa_esp32_main/ilyasa_esp32_main.ino` again (Arduino IDE, board "ESP32 Dev Module").
   For live testing set `LIVE_SEND_MS = 5000UL` near the top before flashing.

## 2. Run it (two terminals, both in `scripts`)
Terminal 1, the live control loop (pick what is plugged in: lamp, fan or geyser):

    python live_control_fast.py --appliance lamp

Terminal 2, the ML dashboard (opens in the browser):

    streamlit run dashboard.py

Open the **Today: power & price** tab. Each appliance gets two graphs:
- **Power**: actual (blue), historical average day (grey dashed), LSTM expected (orange)
- **Price**: cost in RM/hour for the same three, plus the price in sen/kWh (purple). Grey band = ToU peak 2pm-10pm.

Actual readings appear for the appliance you are running. The others show historical + LSTM only.

## 3. Blynk (already set up, nothing to create)
- New widgets: ToU switch (V19), Tariff period (V16), Price now (V17), Cost/hour (V18), Cost today (V20),
  and the **Pricing history** chart (V17 + V18).
- Power history was empty because the datastreams were set to "Last Value" (no history kept).
  They are now "1 Minute AVG", so the charts fill from now on. Older data cannot be recovered.
- Controls only work on **Devices > Ilyasa Energy**, not on the template page.

## 4. Quick checks
- No waiting: `python demo_decision_now.py --tou-scenario fan_peak --push` (LCD should show `SHIFT  PEAK`).
- Normal tariff instead of ToU: turn the Blynk ToU switch off, or add `--tariff a`.

Note: ToU savings only apply if the house is enrolled in TNB ToU.
