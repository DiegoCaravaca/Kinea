from pathlib import Path
import time
import csv
import numpy as np
from datetime import datetime
from gpiozero import MCP3008

# Resolve paths from this file so the script runs from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Capture settings
CHANNEL = 0
SAMPLE_RATE = 200                                       # Samples per second
WINDOW_MS = 250                                         # Window length
SAMPLES_PER_WINDOW = int(SAMPLE_RATE * WINDOW_MS / 1000)  # 50 samples
RECORD_SECONDS = 10                                     # Time per repetition
REPETITIONS = 3                                         # Repetitions per gesture
VREF = 3.3                                              # ADC reference voltage

GESTURES = {
    0: "CLOSED FIST",
    1: "OPEN HAND",
}

sensor = MCP3008(channel=CHANNEL)


def read_voltage():
    """Read one sample and scale it to volts.

    Returns:
        voltage: (float) sensor voltage in the 0..VREF range.
    """
    return sensor.value * VREF


def collect_window():
    """Collect a full window of samples at a fixed rate.

    Returns:
        samples: (list) raw voltage samples for one window.
    """
    samples = []
    interval = 1.0 / SAMPLE_RATE
    for _ in range(SAMPLES_PER_WINDOW):
        samples.append(read_voltage())
        time.sleep(interval)
    return samples


def extract_features(window):
    """Compute five time-domain EMG features from one window.

    Args:
        window: (list) raw voltage samples.

    Returns:
        features: (dict) rms, mav, zc, wl and var for the window.
    """
    w = np.array(window)
    centered = w - np.mean(w)  # Remove DC offset before counting crossings
    return {
        "rms": round(np.sqrt(np.mean(w ** 2)), 6),
        "mav": round(np.mean(np.abs(w)), 6),
        "zc": int(np.sum(np.diff(np.sign(centered)) != 0)),
        "wl": round(np.sum(np.abs(np.diff(w))), 6),
        "var": round(np.var(w), 6),
    }


def monitor_live(duration=8):
    """Show a live RMS reading so the user can check the electrodes.

    Args:
        duration: (int) seconds to keep monitoring.
    """
    print("\n  MONITOR - Open and close your hand to see it change")
    start = time.time()
    while time.time() - start < duration:
        f = extract_features(collect_window())
        bar_len = int(f["rms"] * 30 / VREF)
        bar_str = "#" * bar_len + "." * (30 - bar_len)
        print(f"\r  RMS: {f['rms']:.4f}V  MAV: {f['mav']:.4f}V  "
              f"ZC: {f['zc']:3d}  [{bar_str}]", end="")
    print("\n")


def record_gesture(gesture_id, gesture_name, rep, all_data):
    """Record one repetition of a gesture and append its windows.

    Args:
        gesture_id: (int) label for the gesture (0 closed, 1 open).
        gesture_name: (str) human-readable gesture name.
        rep: (int) zero-based repetition index.
        all_data: (list) list of feature rows, extended in place.
    """
    print(f"\n  Gesture: {gesture_name} - Repetition {rep + 1}/{REPETITIONS}")
    input("  Press ENTER when you are ready...")
    print(f"  >>> NOW! Hold the gesture for {RECORD_SECONDS} seconds <<<")
    time.sleep(0.5)

    windows_count = int(RECORD_SECONDS * 1000 / WINDOW_MS)

    for w in range(windows_count):
        features = extract_features(collect_window())
        features["gesture_id"] = gesture_id
        features["gesture_name"] = gesture_name
        features["repetition"] = rep + 1
        features["window"] = w + 1
        features["timestamp"] = round(time.time(), 3)
        all_data.append(features)

        progress = int((w + 1) / windows_count * 20)
        bar_str = "#" * progress + "." * (20 - progress)
        print(f"\r  Recording [{bar_str}] {w + 1}/{windows_count}", end="")

    print(f"\n  Done")


def main():
    print("=" * 50)
    print("  KINEA - EMG data collection")
    print("  Gestures: CLOSED FIST / OPEN HAND")
    print("=" * 50)

    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = DATA_DIR / f"emg_data_{timestamp}.csv"

    # Check the electrodes pick up a signal before recording
    print("\n  Checking signal...")
    input("  Put on the electrodes and press ENTER...")
    monitor_live(duration=8)

    proceed = input("  Do you see variation? (y/n): ")
    if proceed.lower() != "y":
        print("  Check the connections.")
        return

    # Record every gesture, several repetitions each
    all_data = []
    for gesture_id, gesture_name in GESTURES.items():
        print(f"\n{'=' * 50}")
        print(f"  GESTURE: {gesture_name}")
        print(f"{'=' * 50}")
        for rep in range(REPETITIONS):
            record_gesture(gesture_id, gesture_name, rep, all_data)

    # Write the dataset
    fieldnames = ["timestamp", "gesture_id", "gesture_name",
                  "repetition", "window", "rms", "mav", "zc", "wl", "var"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"\n  Saved to: data/{csv_path.name}")
    print(f"  Total windows: {len(all_data)}")
    print("  Now run: python3 train_model.py")


if __name__ == "__main__":
    main()
