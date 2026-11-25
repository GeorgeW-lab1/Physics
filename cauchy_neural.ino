

const uint8_t SENSOR_PIN = A0;
const unsigned long SAMPLE_INTERVAL_MS = 100;  // 10 Hz
const unsigned long BAUD_RATE = 115200;

unsigned long lastSampleMs = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  analogReference(DEFAULT);  // Uno default: approx. 0–5 V reference
  pinMode(SENSOR_PIN, INPUT);
  delay(300);
  Serial.println(F("MOISTURE_CAUCHY,1"));
  Serial.println(F("millis,raw_adc,voltage_v"));
}

void loop() {
  const unsigned long now = millis();

  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;

    const int raw = analogRead(SENSOR_PIN);  // 0..1023
    const float voltage = raw * (5.0f / 1023.0f);

    Serial.print(now);
    Serial.print(',');
    Serial.print(raw);
    Serial.print(',');
    Serial.println(voltage, 4);
  }
}
