#!/bin/bash
# Install MQTT tracking dependencies and run a quick broker check.

set -e
cd "$(dirname "$0")"

echo "MQTT tracking setup"
if [ ! -d ".venv" ]; then
    echo "Run bash setup.sh first."
    exit 1
fi

source .venv/bin/activate
python -m pip install -q paho-mqtt
python verify.py
echo ""
echo "Optional hardware checks:"
echo "  python test_simple_tracking.py"
echo "  python test_mqtt_system.py"
echo "  python track.py"
