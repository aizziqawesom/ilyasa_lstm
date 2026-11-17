/*
 * Ilyasa PSM2 - main ESP32 sketch. REPLACES relay_control.ino.
 *
 * Written from scratch because the original PSM1 sketch wasn't available.
 * Does everything in one file:
 *   - reads the PZEM-004T (voltage, current, power, energy)
 *   - sends readings to Blynk (within the FREE plan message limit, see below)
 *   - receives the control decision from live_control_loop.py (V10) and
 *     drives the relay (only CUTOFF actually switches the relay off)
 *   - buzzer + 16x2 LCD status
 *   - fail-safe if the cloud connection or the Python script goes quiet
 *
 * Libraries (Arduino IDE > Library Manager):
 *   - "Blynk" by Volodymyr Shymanskyy
 *   - "PZEM004Tv30" by Jakub Mandula
 *   - "LiquidCrystal I2C" by Frank de Brabander
 *   plus the ESP32 board package (Boards Manager > "esp32" by Espressif).
 *
 * PINS
 *   GPIO 26 -> relay IN        GPIO 27 -> buzzer +
 *   GPIO 16 (RX2) <- PZEM TX   GPIO 17 (TX2) -> PZEM RX
 *   GPIO 21 -> LCD SDA         GPIO 22 -> LCD SCL
 *   5V/VIN -> relay VCC, PZEM 5V, LCD VCC     GND -> all GND
 *
 * BLYNK DATASTREAMS (create these in the template, all Virtual Pin):
 *   V0  power_w       Double   0-3000   live power, sent every 5 min
 *   V3  energy_kwh    Double   0-99999  PZEM kWh counter, every 15 min
 *   V5  power_avg_w   Double   0-3000   15-min average power (Python reads this)
 *   V10 control       Integer  0-3      0 OK, 1 NOTIFY, 2 SHIFT, 3 CUTOFF (Python writes)
 *   V11 reason        String            reason text (Python writes)
 *   V12 relay_state   Integer  0-1      relay feedback from this board
 *   V13 manual_mode   Integer  0-2      0 AUTO (follow Python), 1 FORCE ON, 2 FORCE OFF
 *                                       (Segmented Switch widget; overrides V10 until set back to 0)
 *   V14 buzzer_mode   Integer  0-2      0 AUTO (normal beeps), 1 ON (continuous, for testing), 2 MUTE
 *   V15 scaling       Integer  0-1      wattage mapping on/off, read by the PYTHON loop (not by this board)
 *   V6  temperature_c Double   -10-60   DHT22, every 5 min
 *   V7  humidity_pct  Double   0-100    DHT22, every 5 min
 *   V16 tou_period    String            e.g. "PEAK til 10pm" (Python writes; shown on LCD row 1)
 *   V17 tou_rate_sen  Double   0-100    all-in price now, sen/kWh (Python writes)
 *   V18 cost_rm_hour  Double   0-5      cost of the current load per hour, RM (Python writes)
 *   V19 tariff_mode   Integer  0-1      0 normal tariff, 1 TNB ToU (Switch widget, read by Python)
 *   V20 cost_today_rm Double   0-100    running cost today while Python runs, RM (Python writes)
 *
 * DHT22: data pin GPIO 4, VCC 3.3V, GND. Library: "DHT sensor library" by
 * Adafruit (also installs "Adafruit Unified Sensor"). A bare 4-pin DHT22
 * needs a 10k resistor from data to 3.3V; the 3-pin module has it built in.
 *
 * MESSAGE BUDGET: Blynk FREE plan is about 30,000 messages/month. This
 * sketch sends about 400/day, the Python loop about 290/day, roughly
 * 21,000/month total. Do NOT shorten the send intervals below without
 * checking the usage counter in the Blynk console, or the account will hit
 * its limit mid-deployment.
 *
 * FAIL-SAFE (fail-open, relay ON = appliance works normally):
 *   - cloud connection lost for more than 5 min -> relay ON
 *   - no new command from Python for more than 45 min (3 missed 15-min
 *     cycles, e.g. laptop asleep) -> relay ON
 *   Note: the old relay_control.ino used a 5-min "no message" timeout. That
 *   was a bug once Python only writes every 15 min: it would have switched
 *   the heater back ON 5 min after every cutoff. Fixed here.
 *   After a reboot the relay starts ON and waits for the next Python command
 *   (the last V10 value is deliberately NOT re-synced from the server, so a
 *   stale CUTOFF can't come back after a restart).
 */

// ---- From Blynk console > your device > Device info ----
#define BLYNK_TEMPLATE_ID   "TMPLxxxxxxxxx"
#define BLYNK_TEMPLATE_NAME "Ilyasa Energy"
#define BLYNK_AUTH_TOKEN    "YOUR_BLYNK_AUTH_TOKEN"
#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <PZEM004Tv30.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>

const char WIFI_SSID[] = "YOUR_WIFI_SSID";
const char WIFI_PASS[] = "YOUR_WIFI_PASSWORD";

// ---- Pins ----
const int RELAY_PIN   = 26;
const int BUZZER_PIN  = 27;
const int DHT_PIN     = 4;    // DHT22 data
const int PZEM_RX_PIN = 16;   // ESP32 receives on this pin (wire to PZEM TX)
const int PZEM_TX_PIN = 17;   // ESP32 transmits on this pin (wire to PZEM RX)

// ---- Hardware options ----
const bool    RELAY_ACTIVE_HIGH = true;   // if the relay is ON when it should be OFF, set false
// For 5V "active-low" relay boards that stay ON even when the ESP32 pin goes
// to 3.3V (3.3V isn't high enough to switch them off): set RELAY_ACTIVE_HIGH
// = false AND RELAY_OPEN_DRAIN = true. OFF then releases the pin (INPUT) so
// the board's own pull-up switches it off; ON pulls the pin to GND.
// The cleaner hardware fix is the JD-VCC jumper (see BUILD notes).
const bool    RELAY_OPEN_DRAIN = false;
const uint8_t LCD_ADDR = 0x27;            // if the LCD stays blank, try 0x3F

// ---- Timing ----
const unsigned long SAMPLE_MS        = 2000UL;               // read PZEM
const unsigned long LIVE_SEND_MS     = 5UL * 60UL * 1000UL;  // V0
const unsigned long AVG_SEND_MS      = 15UL * 60UL * 1000UL; // V5 + V3
const unsigned long CLOUD_LOSS_MS    = 5UL * 60UL * 1000UL;
const unsigned long STALE_COMMAND_MS = 45UL * 60UL * 1000UL;
const unsigned long WARNING_MS       = 30UL * 1000UL;        // buzzer warning before CUTOFF
const bool FAILSAFE_RELAY_ON = true;
const bool SEND_ENERGY = true;   // set false if the Blynk template won't allow a 6th datastream
const bool SERIAL_DEBUG = true;  // prints PZEM readings, relay and Blynk events to Serial Monitor (115200 baud)
// For the first 10 min after power-up, send live power every 10 s so the
// dashboard shows something straight away while testing (about 60 extra
// messages per boot). After that it drops to the normal every-5-min rate.
const unsigned long FAST_SEND_WINDOW_MS = 10UL * 60UL * 1000UL;
const unsigned long FAST_SEND_MS = 10000UL;

PZEM004Tv30 pzem(Serial2, PZEM_RX_PIN, PZEM_TX_PIN);
LiquidCrystal_I2C lcd(LCD_ADDR, 16, 2);
DHT dht(DHT_PIN, DHT22);
BlynkTimer timer;

float lastV = NAN, lastI = NAN, lastP = NAN, lastE = NAN;
double powerSum = 0.0;
unsigned long powerCount = 0;

int controlValue = 0;
bool haveCommand = false;
unsigned long lastCommandAt = 0;
unsigned long lastCloudOkAt = 0;
bool relayIsOn = true;
bool warningActive = false;
unsigned long warningStartedAt = 0;
bool failsafeActive = false;
String reasonText = "";
String touShort = "";   // "PEAK" / "OFFPK" from V16, blank on the normal tariff
bool lcdShowReason = false;
int manualMode = 0;
int buzzerMode = 0;   // 0 AUTO (normal beeps), 1 ON (continuous), 2 MUTE (V14)
float lastTempC = NAN, lastHum = NAN;   // 0 AUTO (follow Python / V10), 1 FORCE ON, 2 FORCE OFF (V13)
unsigned long pzemFailCount = 0;

void relayWrite(bool on) {
  relayIsOn = on;
  if (RELAY_OPEN_DRAIN) {
    bool drive = (on != RELAY_ACTIVE_HIGH);   // active-low: pull LOW to switch on
    if (drive) {
      pinMode(RELAY_PIN, OUTPUT);
      digitalWrite(RELAY_PIN, LOW);
    } else {
      pinMode(RELAY_PIN, INPUT);              // let go: board pull-up switches it off
    }
  } else {
    digitalWrite(RELAY_PIN, (on == RELAY_ACTIVE_HIGH) ? HIGH : LOW);
  }
  if (Blynk.connected()) Blynk.virtualWrite(V12, on ? 1 : 0);
  if (SERIAL_DEBUG) Serial.printf("[RELAY] %s\n", on ? "ON (load powered)" : "OFF (load cut)");
}

void buzz(bool on) {
  // every buzzer write goes through here, so the manual buzzer mode always wins
  bool level = (buzzerMode == 1) ? true : (buzzerMode == 2) ? false : on;
  digitalWrite(BUZZER_PIN, level ? HIGH : LOW);
}

void beep(int times) {
  for (int k = 0; k < times; k++) {
    buzz(true);
    delay(120);
    buzz(false);
    delay(120);
  }
}

void applyCommand(int v) {
  controlValue = v;
  if (manualMode != 0) {   // manual override: remember the command, but don't touch the relay
    if (SERIAL_DEBUG) Serial.printf("[CONTROL] V10=%d ignored for now: manual mode is ON (set V13 to 0 for AUTO)\n", v);
    return;
  }
  switch (v) {
    case 0:  // OK
      warningActive = false;
      buzz(false);
      relayWrite(true);
      break;
    case 1:  // NOTIFY
      warningActive = false;
      relayWrite(true);
      beep(1);
      break;
    case 2:  // SHIFT (fan policy; relay stays on)
      warningActive = false;
      relayWrite(true);
      beep(2);
      break;
    case 3:  // CUTOFF: warn first, cut after WARNING_MS (see checkSafety)
      if (relayIsOn && !warningActive) {
        warningActive = true;
        warningStartedAt = millis();
        if (SERIAL_DEBUG) Serial.println("[CONTROL] CUTOFF requested: 30 s buzzer warning, then relay OFF");
      }
      break;
    default:
      break;
  }
}

BLYNK_WRITE(V10) {
  haveCommand = true;
  lastCommandAt = millis();
  failsafeActive = false;
  int v = param.asInt();
  if (SERIAL_DEBUG) Serial.printf("[BLYNK] V10 control = %d (0 OK, 1 NOTIFY, 2 SHIFT, 3 CUTOFF)\n", v);
  applyCommand(v);
}

BLYNK_WRITE(V13) {
  manualMode = param.asInt();
  warningActive = false;
  buzz(false);
  if (SERIAL_DEBUG) Serial.printf("[MANUAL] V13 = %d (%s)\n", manualMode,
                                  manualMode == 1 ? "FORCE ON" : manualMode == 2 ? "FORCE OFF" : "AUTO");
  if (manualMode == 1) {
    relayWrite(true);
  } else if (manualMode == 2) {
    relayWrite(false);   // manual OFF is immediate, no 30 s warning: you pressed it yourself
  } else {
    manualMode = 0;
    applyCommand(controlValue);   // back to AUTO: re-apply the last command from Python
  }
}

BLYNK_WRITE(V14) {
  buzzerMode = param.asInt();
  if (buzzerMode < 0 || buzzerMode > 2) buzzerMode = 0;
  buzz(false);   // applies the new mode straight away (ON sounds, AUTO/MUTE go quiet)
  if (SERIAL_DEBUG) Serial.printf("[MANUAL] V14 buzzer = %d (%s)\n", buzzerMode,
                                  buzzerMode == 1 ? "ON" : buzzerMode == 2 ? "MUTE" : "AUTO");
}

BLYNK_WRITE(V15) {
  // The mapping is applied by the PYTHON loop, not by this board. This only
  // confirms the switch press reached the ESP32 (and shows it on the LCD).
  if (SERIAL_DEBUG) Serial.printf("[BLYNK] V15 wattage mapping = %s (applied by the Python loop on its next cycle)\n",
                                  param.asInt() ? "ON" : "OFF");
  reasonText = param.asInt() ? "Mapping: ON" : "Mapping: OFF";
}

BLYNK_WRITE(V16) {
  String s = param.asStr();
  touShort = s.startsWith("PEAK") ? "PEAK" : s.startsWith("OFF") ? "OFFPK" : "";
  if (SERIAL_DEBUG) Serial.printf("[BLYNK] V16 tariff period = %s\n", s.c_str());
}

BLYNK_WRITE(V11) {
  reasonText = param.asStr();
  if (SERIAL_DEBUG) Serial.printf("[BLYNK] V11 reason = %s\n", reasonText.c_str());
}

BLYNK_CONNECTED() {
  lastCloudOkAt = millis();
  Blynk.virtualWrite(V12, relayIsOn ? 1 : 0);
  Blynk.virtualWrite(V13, manualMode);
  Blynk.virtualWrite(V14, buzzerMode);   // after a reboot this shows AUTO (0) on the dashboard
  Blynk.syncVirtual(V16);                // get the current ToU period back after a reboot
  if (SERIAL_DEBUG) Serial.println("[BLYNK] connected to cloud");
}

void samplePzem() {
  // Read every field separately, so voltage still shows even if power fails
  float v  = pzem.voltage();
  float i  = pzem.current();
  float p  = pzem.power();
  float e  = pzem.energy();
  float f  = pzem.frequency();
  float pf = pzem.pf();
  lastV = v;
  lastI = i;
  lastE = e;
  if (!isnan(p)) {
    lastP = p;
    powerSum += p;
    powerCount++;
    pzemFailCount = 0;
  } else {
    lastP = NAN;
    pzemFailCount++;
  }

  if (!SERIAL_DEBUG) return;
  if (isnan(v) && isnan(p)) {
    Serial.printf("[PZEM] NO RESPONSE (%lu in a row). Check: 240V reaching PZEM L/N terminals "
                  "(it needs mains to answer), PZEM TX -> GPIO %d, PZEM RX -> GPIO %d, PZEM 5V and GND\n",
                  pzemFailCount, PZEM_RX_PIN, PZEM_TX_PIN);
    return;
  }
  Serial.printf("[PZEM] V=%.1f V  I=%.3f A  P=%.1f W  E=%.3f kWh  f=%.1f Hz  PF=%.2f  relay=%s\n",
                v, i, p, e, f, pf, relayIsOn ? "ON" : "OFF");
  if (!isnan(v) && v > 100 && !isnan(p) && p < 0.5 && relayIsOn) {
    Serial.println("       (power ~0: nothing plugged in, or load below PZEM minimum ~0.4 W / 0.02 A, e.g. a 3 W LED)");
  }
}

void sendLive() {
  if (Blynk.connected() && !isnan(lastP)) {
    Blynk.virtualWrite(V0, lastP);
    if (SERIAL_DEBUG) Serial.printf("[BLYNK] sent V0 live power = %.1f W\n", lastP);
  }
}

void sampleDht() {
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  lastTempC = t;
  lastHum = h;
  if (!SERIAL_DEBUG) return;
  if (isnan(t) || isnan(h)) {
    Serial.printf("[DHT22] no reading. Check: data -> GPIO %d, VCC -> 3.3V, GND; bare sensor needs 10k to 3.3V\n", DHT_PIN);
  } else {
    Serial.printf("[DHT22] T=%.1f C  RH=%.1f %%\n", t, h);
  }
}

void sendDht() {
  if (Blynk.connected() && !isnan(lastTempC) && !isnan(lastHum)) {
    Blynk.virtualWrite(V6, lastTempC);
    Blynk.virtualWrite(V7, lastHum);
    if (SERIAL_DEBUG) Serial.printf("[BLYNK] sent V6 temp = %.1f C, V7 humidity = %.1f %%\n", lastTempC, lastHum);
  }
}

void sendLiveFastAtStartup() {
  if (millis() < FAST_SEND_WINDOW_MS) {
    sendLive();
    sendDht();
  }
}

void sendAverage() {
  if (!Blynk.connected()) return;   // keep accumulating until we can send
  if (powerCount > 0) {
    Blynk.virtualWrite(V5, powerSum / (double)powerCount);
    if (SERIAL_DEBUG) Serial.printf("[BLYNK] sent V5 15-min average = %.1f W (%lu samples)\n",
                                    powerSum / (double)powerCount, powerCount);
    powerSum = 0.0;
    powerCount = 0;
  }
  if (SEND_ENERGY && !isnan(lastE)) Blynk.virtualWrite(V3, lastE);
}

void checkSafety() {
  unsigned long now = millis();
  if (Blynk.connected()) lastCloudOkAt = now;

  bool cloudLost = (now - lastCloudOkAt) > CLOUD_LOSS_MS;
  bool stale = haveCommand && (now - lastCommandAt) > STALE_COMMAND_MS;
  if ((cloudLost || (stale && manualMode == 0)) && !failsafeActive) {
    failsafeActive = true;
    manualMode = 0;   // losing the cloud cancels manual mode too: fail-safe wins
    buzzerMode = 0;   // and a forced-on buzzer
    haveCommand = false;
    warningActive = false;
    controlValue = 0;
    buzz(false);
    if (SERIAL_DEBUG) Serial.printf("[FAILSAFE] %s -> relay back ON\n",
                                    cloudLost ? "cloud connection lost > 5 min" : "no command from Python > 45 min");
    relayWrite(FAILSAFE_RELAY_ON);
  }

  if (warningActive) {
    unsigned long elapsed = now - warningStartedAt;
    if (elapsed < WARNING_MS) {
      buzz((elapsed / 500UL) % 2UL == 0UL);
    } else {
      buzz(false);
      warningActive = false;
      relayWrite(false);
    }
  }
}

void keepConnected() {
  if (SERIAL_DEBUG) {
    Serial.printf("[NET] WiFi=%s  Blynk=%s  IP=%s\n",
                  WiFi.status() == WL_CONNECTED ? "connected" : "NOT connected",
                  Blynk.connected() ? "connected" : "NOT connected",
                  WiFi.localIP().toString().c_str());
  }
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
    return;
  }
  if (!Blynk.connected()) Blynk.connect(3000);
}

void lcdLine(int row, const char *text) {
  char buf[17];
  snprintf(buf, sizeof(buf), "%-16s", text);
  lcd.setCursor(0, row);
  lcd.print(buf);
}

void updateLcd() {
  char line[32];
  if (failsafeActive) {
    snprintf(line, sizeof(line), "FAILSAFE: ON");
  } else if (manualMode == 1) {
    snprintf(line, sizeof(line), "MANUAL: ON");
  } else if (manualMode == 2) {
    snprintf(line, sizeof(line), "MANUAL: OFF");
  } else if (warningActive) {
    unsigned long left = (WARNING_MS - (millis() - warningStartedAt)) / 1000UL;
    snprintf(line, sizeof(line), "CUTOFF in %lus", left);
  } else if (!relayIsOn) {
    snprintf(line, sizeof(line), "CUTOFF - OFF");
  } else {
    const char *names[] = {"OK", "NOTIFY", "SHIFT", "CUTOFF"};
    const char *n = (controlValue >= 0 && controlValue <= 3) ? names[controlValue] : "?";
    if (touShort.length() > 0) snprintf(line, sizeof(line), "%-7s%s", n, touShort.c_str());   // e.g. "SHIFT  PEAK"
    else snprintf(line, sizeof(line), "Status: %s", n);
  }
  char row0[17];
  snprintf(row0, sizeof(row0), "%-15.15s%c", line, Blynk.connected() ? '*' : '!');
  lcd.setCursor(0, 0);
  lcd.print(row0);

  lcdShowReason = !lcdShowReason;
  if (lcdShowReason && reasonText.length() > 0) {
    lcdLine(1, reasonText.substring(0, 16).c_str());
  } else if (isnan(lastP) && isnan(lastV)) {
    lcdLine(1, "PZEM no reading");
  } else if (isnan(lastP)) {
    snprintf(line, sizeof(line), "%.0fV P:no read", lastV);
    lcdLine(1, line);
  } else {
    if (!isnan(lastTempC)) snprintf(line, sizeof(line), "%.1fW %.0fV %.1fC", lastP, lastV, lastTempC);
    else snprintf(line, sizeof(line), "%.1fW %.0fV %.2fA", lastP, lastV, lastI);
    lcdLine(1, line);
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("=== Ilyasa PSM2 ESP32 ===");
  Serial.printf("Pins: relay GPIO %d, buzzer GPIO %d, PZEM RX GPIO %d (to PZEM TX), PZEM TX GPIO %d (to PZEM RX), LCD I2C 0x%02X on SDA 21 / SCL 22\n",
                RELAY_PIN, BUZZER_PIN, PZEM_RX_PIN, PZEM_TX_PIN, LCD_ADDR);

  // Set the relay level BEFORE making the pin an output, so it doesn't
  // glitch at boot. Starts ON (fail-open) until Python sends a command.
  if (RELAY_OPEN_DRAIN) {
    relayWrite(true);   // relay starts ON (fail-open)
  } else {
    digitalWrite(RELAY_PIN, RELAY_ACTIVE_HIGH ? HIGH : LOW);
    pinMode(RELAY_PIN, OUTPUT);
  }
  relayIsOn = true;
  pinMode(BUZZER_PIN, OUTPUT);
  buzz(false);

  lcd.init();
  lcd.backlight();
  dht.begin();
  lcdLine(0, "Ilyasa PSM2");
  lcdLine(1, "Connecting WiFi");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Blynk.config(BLYNK_AUTH_TOKEN);

  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000UL) delay(250);
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[NET] WiFi connected, IP %s. Connecting to Blynk...\n", WiFi.localIP().toString().c_str());
    Blynk.connect(5000);
  } else {
    Serial.println("[NET] WiFi NOT connected after 15 s: check WIFI_SSID / WIFI_PASS (2.4 GHz network only). Will keep retrying.");
  }
  lastCloudOkAt = millis();

  timer.setInterval(SAMPLE_MS, samplePzem);
  timer.setInterval(10000L, sampleDht);          // DHT22 can't be read faster than every 2 s
  timer.setInterval(LIVE_SEND_MS, sendDht);
  timer.setInterval(LIVE_SEND_MS, sendLive);
  timer.setInterval(FAST_SEND_MS, sendLiveFastAtStartup);
  timer.setInterval(AVG_SEND_MS, sendAverage);
  timer.setInterval(250L, checkSafety);
  timer.setInterval(1500L, updateLcd);
  timer.setInterval(15000L, keepConnected);
}

void loop() {
  if (Blynk.connected()) Blynk.run();
  timer.run();
}
