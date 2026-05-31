from pathlib import Path
import threading
from flask import Flask, render_template, jsonify
from hand import Hand
from predictor import Predictor
from reader import Reader

# Resolve paths from this file so the app runs from any working directory
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)

# Hardware and signal settings
CONFIG = {
    "CHANNEL": 0,    # MCP3008 channel for the EMG sensor
    "VREF": 3.3,     # ADC reference voltage
    "STABILITY": 3,  # Equal predictions needed before moving the hand
    "PIN": 15,       # GPIO pin for the servo
}

# Shared state between the hardware thread and the web layer
telemetry = {
    "rms": 0.0,
    "prediction": "WAITING",
    "state": "UNKNOWN",
}


def hardware_loop():
    """Read EMG, predict the gesture and drive the hand.

    Runs in a background thread so it never blocks Flask. Updates the
    shared `telemetry` dict on every window.
    """
    hand = Hand(CONFIG["PIN"], 45, 135)
    reader = Reader(CONFIG["CHANNEL"], CONFIG["VREF"], 200, 50)
    model = Predictor(MODELS_DIR / "emg_model.pkl",
                      MODELS_DIR / "emg_scaler.pkl",
                      CONFIG["STABILITY"])

    hand.open()

    while True:
        features = reader.extract_features()
        prediction = model.predict(features)
        stable = model.stable_prediction()

        if stable == 0:
            hand.close()
        elif stable == 1:
            hand.open()

        # Publish the latest values for the web layer
        telemetry["rms"] = round(float(features[0]), 4)
        telemetry["prediction"] = "OPEN" if prediction == 1 else "CLOSED"
        telemetry["state"] = "OPEN" if hand.get_hand_state() else "CLOSED"


@app.route('/')
def index():
    """Serve the dashboard page."""
    return render_template('index.html')


@app.route('/data')
def data():
    """Return the current telemetry as JSON.

    Returns:
        response: (Response) JSON with rms, prediction and state.
    """
    return jsonify(telemetry)


if __name__ == "__main__":
    # Start the hardware before serving the web
    threading.Thread(target=hardware_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
