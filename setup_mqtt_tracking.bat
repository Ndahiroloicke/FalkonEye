@echo off
REM Windows helper: MQTT face-tracking + ESP8266 servo setup checks

echo.
echo ======================================================================
echo  FaceLocking MQTT Tracking Setup (Windows)
echo ======================================================================
echo.

cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ERROR: Virtual environment missing. Run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [1/4] Your PC LAN IP addresses (use one as MQTT broker if running Mosquitto locally):
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo   %%a
echo.

echo [2/4] Current MQTT config in src\config.py:
python -c "from src import config; print(f'  Broker: {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}')"
echo.

echo [3/4] Testing MQTT broker connection...
python test_mqtt_system.py
if errorlevel 1 (
    echo.
    echo MQTT test failed. Before running track.py:
    echo   1. Flash arduino\esp8266_camera_tracker\ with your WiFi + broker IP
    echo   2. ESP8266 and this PC must reach the SAME MQTT broker
    echo   3. Update MQTT_BROKER_HOST in src\config.py if needed
    echo   4. Servo: signal=D4, VCC=VIN, GND=GND
    pause
    exit /b 1
)

echo.
echo [4/4] Ready. Start face lock + servo tracking with:
echo   python track.py
echo.
echo Controls: k=lock person  l=unlock  c=center  s=search  q=quit
pause
