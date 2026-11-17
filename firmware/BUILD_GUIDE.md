# Relay Control Module — Hands-On Build Guide

> **Update:** flash `firmware/ilyasa_esp32_main/ilyasa_esp32_main.ino`, not
> `relay_control.ino`. The new sketch does everything in one file (PZEM
> reading, Blynk, relay, buzzer, LCD, fail-safe) and fixes a fail-safe bug in
> the old one. Its header lists the libraries to install and the Blynk
> datastreams to create. Wherever this guide says `V_CONTROL`/`V_STATUS_TEXT`,
> that is still `V10`/`V11`.

For Ilyasa, since you're doing the physical assembly yourself. Read this
whole document before touching anything, especially the safety section.

## ⚠️ Safety — read this first

The relay switches **mains AC voltage** (240V in Malaysia) on the water
heater's live wire. This is genuinely dangerous if done wrong — 240V AC can
kill. If you are not confident handling mains wiring:

- Do this with your supervisor or a lab technician present, not alone.
- Always disconnect the water heater from the wall / trip the circuit
  breaker before touching any wiring. Never work on live wiring.
- Test everything on a **low-power load first** (e.g. a desk lamp or the
  bulb circuit already in your Ch4 prototype) before connecting the actual
  water heater. Confirm the relay clicks and switches correctly on the safe
  load before going anywhere near mains-voltage appliance wiring.
- If in doubt, stop and ask. It's fine to demo the relay logic on the lamp
  circuit for your viva instead of the real water heater if mains wiring
  isn't something you're comfortable doing solo — the control logic is
  what's being examined, not whether you personally wired mains voltage.

## What you need

| Item | Notes |
|---|---|
| ESP32 (already have) | From the existing PSM1 prototype |
| PZEM-004T sensor (already have) | From the existing PSM1 prototype |
| 16x2 I2C LCD (already have) | From the existing PSM1 prototype |
| Buzzer (already have) | From the existing PSM1 prototype |
| 1x single-channel relay module, 5V trigger | Buy one rated for AT LEAST the water heater's current draw + safety margin. Check the water heater's rated wattage/current on its nameplate first — a typical 2000-3000W heater draws ~8-13A at 240V, so get a relay module rated 10A+ (most cheap 5V relay modules on Shopee/Lazada are rated 10A or 16A, either is fine). |
| Jumper wires | For relay IN/VCC/GND to ESP32 |
| Extension cord or plug+socket-outlet pair | Becomes the "smart socket" box — see mains wiring below. Do NOT cut into the water heater's (or any appliance's) own built-in cord. |
| Mains-rated wire + connectors | For splicing the relay into the extension cord's live wire — use proper insulated wire rated for mains, not jumper wires |
| Inline fuse, 3A or 5A, plug-fitting | Cheap extra protection on the live line before the relay, in case of a wiring mistake |

## Wiring — logic side (ESP32 ↔ Relay), safe to do live

1. Relay module **VCC** → ESP32 **5V** (or **VIN** if your relay board needs 5V and your ESP32 dev board exposes it)
2. Relay module **GND** → ESP32 **GND**
3. Relay module **IN** → ESP32 **GPIO 26** (matches `RELAY_PIN` in `relay_control.ino` — change the constant in the code if you wire it to a different pin)
4. Buzzer, if not already wired from the base prototype, → ESP32 **GPIO 27** (matches `BUZZER_PIN`)

This part is low-voltage (5V/3.3V), safe to wire and test without any mains involved.

Before wiring the relay's IN/VCC/GND, confirm the base PSM1 sketch isn't already using GPIO 26 or 27 for something else (the PZEM UART pins and the LCD's I2C pins are the ones to check against). If it is, change `RELAY_PIN`/`BUZZER_PIN` in `relay_control.ino` to free pins instead of reusing a conflicting one.

## Wiring — mains side (build a standalone relay socket box, not the appliance's own cord)

Do not cut into the water heater's (or any appliance's) built-in power cord. Instead, build a reusable "smart socket" box: a plug on one side, a socket outlet on the other, with the relay spliced into the live wire between them. Plug the box into the wall, then plug whatever appliance you're testing (lamp, kettle, water heater) into the box — nothing about the appliance itself is ever opened or modified, and the same box works for every appliance in turn.

1. Trip the circuit breaker for the socket you're working at, or make sure the extension cord is unplugged. Verify with a multimeter or voltage tester that there's no power before touching anything.
2. On the extension cord / plug+socket assembly, identify the **live (L)** wire — the neutral and earth wires are NOT switched, only live goes through the relay.
3. Cut the live wire partway along its length and connect:
   - The end coming FROM the plug (wall side) → inline fuse → relay **COM** terminal
   - The end going TO the socket outlet (appliance side) → relay **NO** (Normally Open) terminal
   - This means: when the relay is de-energized (OFF), the circuit is open and whatever's plugged in has no power. When energized (ON), power flows through. This matches the firmware's `setRelay(true)` = ON = HIGH = closed circuit.
4. Mount the relay, fuse, and wire joints inside the enclosure (item #11 on the hardware list), with the incoming plug cable and outgoing socket cable each passing through a cable-gland or grommeted hole in the enclosure wall.
5. Double-check all mains connections are fully insulated (heat-shrink or proper terminal blocks) before restoring power. No exposed copper.
6. Restore power only after everything is double-checked and the enclosure is closed up. Test on the lamp load (next section) before plugging in the kettle or water heater.

## Flashing the firmware

1. Open `relay_control.ino` in Arduino IDE.
2. Fill in your Blynk auth token and WiFi credentials at the top (the `auth`, `ssid`, `pass` variables).
3. In your Blynk app/dashboard, create three virtual pins if they don't already exist:
   - `V10` (V_CONTROL) — integer widget, this is where `control_logic.py` writes 0/1/2/3
   - `V11` (V_STATUS_TEXT) — string widget, shows the reason text on your LCD
   - `V12` (V_RELAY_STATE) — read-only, just reflects whether the relay is currently on
4. Flash to the ESP32 as usual (same process you already used for the base sensor-reading sketch).

## Testing procedure — do this BEFORE connecting the real water heater

1. Plug a simple lamp (or the bulb circuit from your Ch4 prototype) into the socket box's output — not the water heater yet.
2. Power on the ESP32, confirm it connects to WiFi and Blynk (check the LCD shows "Connecting WiFi" then normal status).
3. From the Blynk app, manually write to `V10`:
   - Write `0` → lamp should turn ON, LCD shows "Status: OK"
   - Write `1` → lamp stays ON, buzzer beeps once, LCD shows "Status: NOTIFY"
   - Write `3` → LCD shows "CUTOFF WARNING", buzzer beeps continuously for 30 seconds, then the lamp turns OFF and LCD shows "CUTOFF - OFF"
4. Unplug the ESP32's WiFi (or turn off your router) for 5+ minutes and confirm the relay defaults to ON (fail-safe behaviour) and the LCD shows "FAILSAFE MODE".
5. Only once all of this works correctly on the lamp, unplug the lamp and plug the real water heater into the same socket box instead — no rewiring needed, since the box itself never changes.

## If something doesn't work

- Relay doesn't click at all → check IN pin wiring and that the relay module's trigger voltage matches (some modules are active-LOW, meaning you'd need to flip `HIGH`/`LOW` in `setRelay()` in the code — test with a multimeter on the IN pin first).
- LCD doesn't update → check the I2C address (0x27 in the code — yours might be 0x3F, common alternative; there are I2C scanner sketches online if unsure).
- Blynk doesn't connect → double check auth token and WiFi credentials, and that you're using the matching Blynk library version for your ESP32 board setup (same one from your base prototype should work).
- Something else → message Aizziq with the specific error/symptom, happy to help debug remotely even without being on-site.
