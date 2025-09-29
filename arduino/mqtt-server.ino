#include <WiFi.h>
#include <PubSubClient.h>
#include <MAX3010x.h>
#include "filters.h"
#include <ArduinoJson.h>

// WiFi & MQTT
const char* ssid = "SSR";
const char* password = "SSR50428";
const char* mqtt_server = "rpi.local";

WiFiClient espClient;
PubSubClient client(espClient);
MAX30105 sensor;

// Filters
LowPassFilter lpf_red(5.0, 100.0);
LowPassFilter lpf_ir(5.0, 100.0);
HighPassFilter hpf(0.5, 100.0);
Differentiator diff(100.0);
MovingAverageFilter<5> avg_bpm;

// State
bool finger_detected = false;
unsigned long last_heartbeat = 0;
unsigned long last_pulse_publish = 0;
unsigned long last_status_publish = 0;
unsigned long last_sample = 0;
float last_diff = NAN;
bool crossed = false;

// SpO2 calibration
const float kSpO2_A = 1.5958422;
const float kSpO2_B = -34.6596622;
const float kSpO2_C = 112.6898759;

// Statistics for R calculation
float red_min = 999999, red_max = 0, red_sum = 0;
float ir_min = 999999, ir_max = 0, ir_sum = 0;
int stat_count = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("Heart Rate Monitor Starting...");

  // Initialize sensor
  if (!sensor.begin() || !sensor.setSamplingRate(sensor.SAMPLING_RATE_100SPS)) {
    Serial.println("Sensor failed!");
    while(1) delay(1000);
  }
  Serial.println("Sensor OK");

  // Connect WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    yield();
  }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());

  // Setup MQTT
  client.setServer(mqtt_server, 1883);
  Serial.println("Ready!");
}

void resetStats() {
  red_min = ir_min = 999999;
  red_max = ir_max = red_sum = ir_sum = 0;
  stat_count = 0;
}

void updateStats(float red, float ir) {
  red_min = min(red_min, red);
  red_max = max(red_max, red);
  red_sum += red;
  
  ir_min = min(ir_min, ir);
  ir_max = max(ir_max, ir);
  ir_sum += ir;
  
  stat_count++;
}

void reconnectMQTT() {
  if (!client.connected()) {
    Serial.print("MQTT connecting...");
    if (client.connect("ESP32HeartRate")) {
      Serial.println("OK");
    }
  }
}

void publishData(const char* topic, JsonDocument& doc) {
  if (client.connected()) {
    char buffer[300];
    serializeJson(doc, buffer);
    client.publish(topic, buffer);
    Serial.print("Published to ");
    Serial.print(topic);
    Serial.print(": ");
    Serial.println(buffer);
  }
}

void loop() {
  yield(); // Prevent watchdog
  
  unsigned long now = millis();
  
  // Sample at 100Hz
  if (now - last_sample < 10) return;
  last_sample = now;
  
  // Reconnect MQTT if needed
  if (now % 5000 == 0) reconnectMQTT();
  if (client.connected()) client.loop();
  
  // Read sensor
  MAX30105Sample raw = sensor.readSample(1000);
  float red = raw.red;
  float ir = raw.ir;
  
  // Finger detection
  if (red < 10000) {
    if (finger_detected) {
      // Send no finger status (rate limited)
      if (now - last_status_publish > 1000) {
        StaticJsonDocument<100> doc;
        doc["type"] = "status";
        doc["message"] = "no_finger";
        publishData("status/data", doc);
        last_status_publish = now;
        
        Serial.println("Finger removed");
        finger_detected = false;
        resetStats();
        lpf_red.reset();
        lpf_ir.reset();
        hpf.reset();
        diff.reset();
        avg_bpm.reset();
      }
    }
    return;
  }
  
  if (!finger_detected) {
    // Send finger detected status
    StaticJsonDocument<100> doc;
    doc["type"] = "status";
    doc["message"] = "finger_detected";
    publishData("status/data", doc);
    
    Serial.println("Finger detected");
    finger_detected = true;
    resetStats();
  }
  
  // Filter signals
  red = lpf_red.process(red);
  ir = lpf_ir.process(ir);
  updateStats(red, ir);
  
  // Publish pulse data (20Hz max) - Match HTML expectations
  if (now - last_pulse_publish > 50) {
    StaticJsonDocument<150> doc;
    doc["type"] = "pulse";
    doc["ir_value"] = (int)ir;
    doc["red_value"] = (int)red;
    publishData("pulse/data", doc);
    last_pulse_publish = now;
  }
  
  // Heart rate detection
  float filtered = hpf.process(red);
  float current_diff = diff.process(filtered);
  
  if (!isnan(current_diff) && !isnan(last_diff)) {
    // Detect peak
    if (last_diff > 0 && current_diff < 0) {
      crossed = true;
    }
    
    // Detect strong downward slope (heartbeat)
    if (crossed && current_diff < -2000) {
      if (last_heartbeat > 0) {
        unsigned long interval = now - last_heartbeat;
        if (interval > 300 && interval < 2000) { // 30-200 BPM range
          int bpm = 60000 / interval;
          int avg_hr = avg_bpm.process(bpm);
          
          if (avg_bpm.count() >= 3 && stat_count > 50) {
            // Calculate SpO2
            float red_ac = red_max - red_min;
            float red_dc = red_sum / stat_count;
            float ir_ac = ir_max - ir_min;
            float ir_dc = ir_sum / stat_count;
            
            float r = (red_ac / red_dc) / (ir_ac / ir_dc);
            float spo2 = kSpO2_A * r * r + kSpO2_B * r + kSpO2_C;
            spo2 = constrain(spo2, 70, 100);
            
            // Publish results - Match HTML expectations
            StaticJsonDocument<200> doc;
            doc["type"] = "metrics";
            doc["heart_rate"] = avg_hr;
            doc["spo2"] = (int)spo2;
            doc["r_value"] = r;
            doc["timestamp"] = now;
            publishData("metrics/data", doc);
            
            Serial.printf("HR: %d BPM, SpO2: %d%%\n", avg_hr, (int)spo2);
            
            resetStats();
          }
        }
      }
      last_heartbeat = now;
      crossed = false;
    }
  }
  
  last_diff = current_diff;
}
