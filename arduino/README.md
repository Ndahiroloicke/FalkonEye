# Arduino Code for ESP8266 Camera Tracker

This directory contains the Arduino sketch for ESP8266-based camera tracking with servo motor control.

## 📁 Files

- `esp8266_camera_tracker/esp8266_camera_tracker.ino` - Main Arduino sketch

## 🚀 Quick Start

### 1. Hardware Setup

Connect servo motor to ESP8266:
- Servo Signal → D4 (GPIO2)
- Servo VCC → 5V
- Servo GND → GND

### 2. Software Setup

1. Install Arduino IDE 2.0+
2. Add ESP8266 board support:
   - File → Preferences
   - Add URL: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
   - Tools → Board → Boards Manager → Install "esp8266"
3. Install libraries:
   - Tools → Manage Libraries
   - Install: **PubSubClient** (for MQTT)

### 3. Configure

Edit `esp8266_camera_tracker.ino`:

```cpp
// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT broker (your computer's IP)
const char* mqtt_server = "192.168.1.100";
```

### 4. Upload

1. Connect ESP8266 via USB
2. Select board: Tools → Board → NodeMCU 1.0
3. Select port: Tools → Port → (your ESP8266 port)
4. Click Upload (→)
5. Open Serial Monitor (115200 baud) to verify

## 📡 MQTT Topics

| Topic | Type | Description |
|-------|------|-------------|
| `camera/track/horizontal` | Subscribe | Angle 0-180° |
| `camera/track/command` | Subscribe | left/right/center |
| `camera/status` | Publish | Current position JSON |

## 🔧 Configuration

### Servo Settings

```cpp
const int SERVO_PIN = D4;              // GPIO pin
const int SERVO_MIN_ANGLE = 0;         // Minimum angle
const int SERVO_MAX_ANGLE = 180;       // Maximum angle
const int SERVO_CENTER_ANGLE = 90;     // Center position
const int SERVO_STEP_SIZE = 5;         // Degrees per step
const int MOVEMENT_DELAY = 15;         // ms between steps
```

### MQTT Settings

```cpp
const char* mqtt_server = "192.168.1.100";  // Broker IP
const int mqtt_port = 1883;                  // Broker port
const char* mqtt_user = "";                  // Username (optional)
const char* mqtt_password = "";              // Password (optional)
```

## 🧪 Testing

### Test Servo Movement

Upload this simple test sketch first:

```cpp
#include <Servo.h>

Servo myServo;

void setup() {
  myServo.attach(D4);
  myServo.write(90);  // Center
}

void loop() {
  myServo.write(0);    // Left
  delay(1000);
  myServo.write(90);   // Center
  delay(1000);
  myServo.write(180);  // Right
  delay(1000);
}
```

### Test MQTT Connection

After uploading main sketch, use mosquitto tools:

```bash
# Subscribe to status
mosquitto_sub -h localhost -t "camera/status" -v

# Send commands
mosquitto_pub -h localhost -t "camera/track/command" -m "center"
mosquitto_pub -h localhost -t "camera/track/command" -m "left"
mosquitto_pub -h localhost -t "camera/track/command" -m "right"
mosquitto_pub -h localhost -t "camera/track/horizontal" -m "45"
```

## 🐛 Troubleshooting

### Servo Jitters
- Use external 5V power supply
- Add 100µF capacitor across servo power
- Increase `MOVEMENT_DELAY`

### WiFi Connection Fails
- Check SSID/password
- Use 2.4GHz WiFi (not 5GHz)
- Move closer to router

### MQTT Connection Fails
- Verify broker IP address
- Check broker is running
- Disable firewall temporarily

### ESP8266 Resets
- Servo drawing too much current
- Use external power for servo
- Check USB cable quality

## 📊 Serial Monitor Output

Expected output:
```
=================================
ESP8266 Camera Tracker Starting
=================================

✓ Servo initialized at center position
Connecting to WiFi: YourNetwork
✓ WiFi connected!
  IP address: 192.168.1.150
  Signal strength: -45 dBm
Connecting to MQTT broker... connected!
✓ Subscribed to topics:
  - camera/track/horizontal
  - camera/track/command
Waiting for MQTT commands...

📨 Received [camera/track/command]: left
← Moving left to: 85
✓ Reached target position: 85
```

## 🔌 Pin Reference

### NodeMCU Pin Mapping

| NodeMCU | GPIO | Function |
|---------|------|----------|
| D0 | GPIO16 | - |
| D1 | GPIO5 | I2C SCL |
| D2 | GPIO4 | I2C SDA |
| D3 | GPIO0 | Flash button |
| D4 | GPIO2 | Built-in LED, Servo |
| D5 | GPIO14 | SPI SCK |
| D6 | GPIO12 | SPI MISO |
| D7 | GPIO13 | SPI MOSI |
| D8 | GPIO15 | SPI CS |

## 📚 Additional Resources

- [ESP8266 Arduino Core](https://arduino-esp8266.readthedocs.io/)
- [PubSubClient Library](https://pubsubclient.knolleary.net/)
- [Servo Library Reference](https://www.arduino.cc/reference/en/libraries/servo/)

## 🎯 Next Steps

1. Test servo movement independently
2. Verify WiFi connection
3. Test MQTT communication
4. Integrate with Python face recognition
5. Fine-tune tracking parameters

See `../MQTT_CAMERA_TRACKING.md` for complete setup guide.
