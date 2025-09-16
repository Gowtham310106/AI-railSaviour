#include <ArduinoJson.h>

// Pin definitions
const int redLED = 3;     
const int greenLED = 4;   
const int blueLED = 5;    
const int pumpRelay = 6;  

String inputString = "";
unsigned long pumpStartTime = 0;
const unsigned long pumpDuration = 10000; 
bool pumpRunning = false;

void setup() {
  Serial.begin(9600);
  pinMode(redLED, OUTPUT);
  pinMode(greenLED, OUTPUT);
  pinMode(blueLED, OUTPUT);
  pinMode(pumpRelay, OUTPUT);  // NEW: Pump relay pin
  
  digitalWrite(redLED, LOW);
  digitalWrite(greenLED, LOW);
  digitalWrite(blueLED, LOW);
  digitalWrite(pumpRelay, HIGH);  
  
  Serial.println("Arduino ready with pump control");
}

void loop() {
  // Handle serial communication
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    if (inChar == '\n') {
      handleCommand(inputString);
      inputString = "";
    }
  }
  
  // Handle pump auto-shutoff
  if (pumpRunning && (millis() - pumpStartTime >= pumpDuration)) {
    stopPump();
  }
}

void handleCommand(String jsonData) {
  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, jsonData);
  
  if (error) {
    Serial.println("❌ JSON parse error");
    return;
  }
  
  String command = doc["command"];
  
  // Turn off all LEDs first (but keep pump state)
  digitalWrite(redLED, LOW);
  digitalWrite(greenLED, LOW);
  if (!pumpRunning) {
    digitalWrite(blueLED, LOW);
  }
  
  if (command == "RED") {
    digitalWrite(redLED, HIGH);
    Serial.println("🔴 Red LED ON - Animal Detected");
  } 
  else if (command == "GREEN") {
    digitalWrite(greenLED, HIGH);
    Serial.println("🟢 Green LED ON - Area Safe");
  } 
  else if (command == "PUMP_ON") {
    startPump();
  } 
  else {
    Serial.println("⚠️ Unknown command");
  }
}

void startPump() {
  if (!pumpRunning) {
    digitalWrite(pumpRelay, LOW);  // Turn ON relay (pump starts)
    digitalWrite(blueLED, HIGH);    // Blue LED indicates pump is running
    pumpRunning = true;
    pumpStartTime = millis();
    Serial.println("💧 Water Pump STARTED");
  } else {
    Serial.println("💧 Pump already running");
  }
}

void stopPump() {
  digitalWrite(pumpRelay, HIGH);   // Turn OFF relay (pump stops)
  digitalWrite(blueLED, LOW);     // Turn off blue LED
  pumpRunning = false;
  Serial.println("💧 Water Pump STOPPED");
}