/*
 * Week 3 deliverable: relay control firmware for the water heater (geyser
 * proxy) channel. Extends the base sensor-reading sketch already in the
 * submitted PSM1 report (Ch3.7.1, Figures 3.4/3.5) with the actuation side
 * of the control loop.
 *
 * ARCHITECTURE (matches Figure 3.2's existing System Block Diagram, which
 * already shows a bidirectional Blynk <-> LSTM arrow):
 *   1. ESP32 reads PZEM-004T sensor, sends readings to Blynk (existing).
 *   2. Python (control_logic.py, run_control_simulation.py) reads recent
 *      history from Blynk, runs the trained LSTM model, computes a control
 *      decision, and WRITES that decision back to a Blynk virtual pin.
 *   3. THIS FIRMWARE reads that virtual pin and drives the relay
 *      accordingly. The ESP32 has no model logic of its own -- it only
 *      acts on what the Python side decided.
 *
 * APPLIANCE POLICY (water heater / geyser proxy -- the only appliance with
 * a HARD relay cutoff; fan gets soft-shift and laptop gets notify-only,
 * both handled purely on the Python/Blynk-notification side, no relay
 * needed for those two):
 *   Control values received on V_CONTROL:
 *     0 = OK      -> relay stays ON  (normal operation)
 *     1 = NOTIFY  -> relay stays ON, buzzer gives 1 short beep (warning only)
 *     2 = SHIFT   -> relay stays ON, buzzer gives 2 short beeps (not used
 *                    for geyser in the current appliance policy, kept for
 *                    completeness/future appliances on a relay)
 *     3 = CUTOFF  -> relay OFF, buzzer gives a longer continuous warning
 *                    for WARNING_SECONDS before actually cutting power
 *                    (pre-emptive warning -- confirmed Week 3 scope: nobody
 *                    wants hot water cut mid-shower with zero warning)
 *
 * FAIL-SAFE ON DISCONNECT (documented decision, override if Ilyasa/
 * supervisor wants the opposite): if Blynk connection is lost for more
 * than FAILSAFE_TIMEOUT_MS, the relay defaults to ON (fail-OPEN --
 * appliance keeps running). Rationale: losing hot water because of a WiFi
 * hiccup is more disruptive to the household than the (bounded, short-term)
 * risk of a missed cutoff during that same disconnect window. If the
 * supervisor prefers fail-CLOSED (safer default, appliance cuts on any
 * disconnect), flip FAILSAFE_DEFAULT_RELAY_STATE below and say so
 * explicitly in Ch3.8 -- this is a real safety-engineering decision that
 * should be stated, not left implicit.
 *
 * HARDWARE (in addition to what's already in the PSM1 prototype):
 *   - 1x single-channel relay module (5V trigger, rated for mains AC load
 *     matching the water heater's current draw -- check the water heater's
 *     rated current before buying inline with the model number)
 *   - Relay IN pin -> ESP32 GPIO (RELAY_PIN below)
 *   - Relay VCC -> 5V, GND -> GND
 *   - Water heater's LIVE wire routed THROUGH the relay's NO (normally
 *     open) contact -- see build guide (BUILD_GUIDE.md) for the wiring
 *     diagram and safety warnings before touching any mains wiring.
 */

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <LiquidCrystal_I2C.h>

// ---- Fill in before flashing ----
char auth[] = "YOUR_BLYNK_AUTH_TOKEN";
char ssid[] = "YOUR_WIFI_SSID";
char pass[] = "YOUR_WIFI_PASSWORD";

// ---- Pin assignments ----
const int RELAY_PIN = 26;       // drives the relay module's IN pin
const int BUZZER_PIN = 27;      // existing buzzer from the base prototype

// ---- Blynk virtual pins ----
#define V_CONTROL      V10      // int, 0=OK 1=NOTIFY 2=SHIFT 3=CUTOFF (written by control_logic.py)
#define V_STATUS_TEXT  V11      // string, human-readable reason (written by control_logic.py)
#define V_RELAY_STATE  V12      // read-only feedback to the dashboard: 1=ON 0=OFF

// ---- Fail-safe config ----
const unsigned long FAILSAFE_TIMEOUT_MS = 5UL * 60UL * 1000UL;  // 5 minutes
const bool FAILSAFE_DEFAULT_RELAY_STATE = true;  // true = fail-OPEN (relay ON / appliance keeps running)

// ---- Pre-emptive warning before actual cutoff ----
const unsigned long WARNING_SECONDS = 30;  // buzzer warns for this long before CUTOFF actually opens the relay

LiquidCrystal_I2C lcd(0x27, 16, 2);

int lastControlValue = 0;          // 0=OK by default until first command arrives
unsigned long lastBlynkMessageAt = 0;
bool relayIsOn = true;
bool cutoffWarningActive = false;
unsigned long cutoffWarningStartedAt = 0;

void setRelay(bool on) {
  relayIsOn = on;
  digitalWrite(RELAY_PIN, on ? HIGH : LOW);  // most relay modules: HIGH = energized = closed circuit = load ON
  Blynk.virtualWrite(V_RELAY_STATE, on ? 1 : 0);
}

void shortBeep(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(150);
    digitalWrite(BUZZER_PIN, LOW);
    delay(150);
  }
}

BLYNK_WRITE(V_CONTROL) {
  lastControlValue = param.asInt();
  lastBlynkMessageAt = millis();

  switch (lastControlValue) {
    case 0:  // OK
      cutoffWarningActive = false;
      setRelay(true);
      lcd.setCursor(0, 0);
      lcd.print("Status: OK      ");
      break;

    case 1:  // NOTIFY
      cutoffWarningActive = false;
      setRelay(true);
      shortBeep(1);
      lcd.setCursor(0, 0);
      lcd.print("Status: NOTIFY  ");
      break;

    case 2:  // SHIFT (not used for geyser's own policy, but handled for completeness)
      cutoffWarningActive = false;
      setRelay(true);
      shortBeep(2);
      lcd.setCursor(0, 0);
      lcd.print("Status: SHIFT   ");
      break;

    case 3:  // CUTOFF -- warn first, then actually cut after WARNING_SECONDS
      if (!cutoffWarningActive) {
        cutoffWarningActive = true;
        cutoffWarningStartedAt = millis();
      }
      lcd.setCursor(0, 0);
      lcd.print("CUTOFF WARNING  ");
      break;
  }
}

BLYNK_WRITE(V_STATUS_TEXT) {
  String reason = param.asStr();
  lcd.setCursor(0, 1);
  lcd.print(reason.substring(0, 16));  // 16x2 LCD, truncate to fit
}

BLYNK_CONNECTED() {
  lastBlynkMessageAt = millis();
  Blynk.syncVirtual(V_CONTROL, V_STATUS_TEXT);
}

void checkFailsafe() {
  if (millis() - lastBlynkMessageAt > FAILSAFE_TIMEOUT_MS) {
    if (relayIsOn != FAILSAFE_DEFAULT_RELAY_STATE) {
      setRelay(FAILSAFE_DEFAULT_RELAY_STATE);
      lcd.setCursor(0, 0);
      lcd.print("FAILSAFE MODE   ");
      lcd.setCursor(0, 1);
      lcd.print(FAILSAFE_DEFAULT_RELAY_STATE ? "Relay: ON (safe)" : "Relay: OFF(safe)");
    }
  }
}

void checkCutoffWarning() {
  if (!cutoffWarningActive) return;

  unsigned long elapsedSec = (millis() - cutoffWarningStartedAt) / 1000UL;
  if (elapsedSec < WARNING_SECONDS) {
    // continuous warning beep pattern while counting down
    if (elapsedSec % 2 == 0) {
      digitalWrite(BUZZER_PIN, HIGH);
    } else {
      digitalWrite(BUZZER_PIN, LOW);
    }
  } else {
    // warning period over, actually cut the relay now
    digitalWrite(BUZZER_PIN, LOW);
    setRelay(false);
    cutoffWarningActive = false;
    lcd.setCursor(0, 0);
    lcd.print("CUTOFF - OFF    ");
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  setRelay(true);  // default ON at boot until first Blynk command arrives

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi ");

  Blynk.begin(auth, ssid, pass);
  lastBlynkMessageAt = millis();
}

void loop() {
  Blynk.run();
  checkFailsafe();
  checkCutoffWarning();
}
