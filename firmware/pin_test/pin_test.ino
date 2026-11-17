/*
 * Relay test: no WiFi, no Blynk. Switches the relay every 2 seconds using
 * the method chosen below, so you can find which one works for your board.
 * Buzzer on GPIO 27 beeps in step with it. Serial Monitor at 115200.
 *
 *   MODE 1 = normal (pin HIGH / LOW)
 *   MODE 2 = open-drain (pin LOW / released) -- for 5V active-low boards
 *            that stay ON with 3.3V
 */
const int MODE = 2;
const int RELAY_PIN = 26;
const int BUZZER_PIN = 27;

void relayDrive(bool low) {
  if (MODE == 2) {
    if (low) { pinMode(RELAY_PIN, OUTPUT); digitalWrite(RELAY_PIN, LOW); }
    else     { pinMode(RELAY_PIN, INPUT); }
  } else {
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, low ? LOW : HIGH);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.printf("Relay test, MODE %d\n", MODE);
}

void loop() {
  relayDrive(true);
  digitalWrite(BUZZER_PIN, HIGH);
  Serial.println("IN pulled LOW  (active-low board: relay should be ON, LED lit)");
  delay(2000);
  relayDrive(false);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println(MODE == 2 ? "IN released    (relay should be OFF, LED off)" : "IN HIGH 3.3V   (relay should be OFF)");
  delay(2000);
}
