import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Resolve paths from this file so the script runs from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def load_data(csv_path):
    """Load the CSV and split features from labels.

    Args:
        csv_path: (str|Path) path to the EMG dataset.

    Returns:
        X: (ndarray) feature matrix, columns [rms, mav, zc, wl, var].
        y: (ndarray) gesture labels (0 closed, 1 open).
        feature_cols: (list) feature column names.
    """
    df = pd.read_csv(csv_path)

    feature_cols = ["rms", "mav", "zc", "wl", "var"]
    X = df[feature_cols].values
    y = df["gesture_id"].values

    print(f"  Data loaded: {csv_path}")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {feature_cols}")
    print(f"  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, feature_cols


def train_and_compare(X, y):
    """Try four classifiers and keep the best one.

    Args:
        X: (ndarray) feature matrix.
        y: (ndarray) gesture labels.

    Returns:
        best_clf: (estimator) classifier refit on all the data.
        scaler: (StandardScaler) scaler fit on the features.
        best_name: (str) name of the chosen classifier.
    """
    # Normalize to mean 0, std 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 80% train, 20% held-out test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = {
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    print("\n" + "=" * 50)
    print("  CLASSIFIER COMPARISON")
    print("  5-fold cross-validation")
    print("=" * 50)

    results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()
        results[name] = (mean_acc, std_acc, clf)

        bar_len = int(mean_acc * 30)
        bar_str = "#" * bar_len + "." * (30 - bar_len)
        print(f"\n  {name}:")
        print(f"    Accuracy: {mean_acc:.3f} (+/- {std_acc:.3f})")
        print(f"    [{bar_str}] {mean_acc * 100:.1f}%")

    # Pick the best by mean accuracy
    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_std, best_clf = results[best_name]

    print(f"\n  >>> BEST: {best_name} with {best_acc * 100:.1f}% <<<")

    # Train the best on the train split, then score the held-out test set
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)

    print(f"\n{'=' * 50}")
    print("  TEST SET EVALUATION")
    print(f"{'=' * 50}")
    print(f"\n  Test accuracy: {best_clf.score(X_test, y_test) * 100:.1f}%")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"                  Predicted")
    print(f"                  Closed   Open")
    print(f"  Real Closed   [ {cm[0][0]:4d}    {cm[0][1]:4d} ]")
    print(f"       Open     [ {cm[1][0]:4d}    {cm[1][1]:4d} ]")

    print(f"\n  Per-class detail:")
    report = classification_report(y_test, y_pred, target_names=["CLOSED", "OPEN"])
    for line in report.split("\n"):
        print(f"    {line}")

    # Refit on all the data for the final model
    best_clf.fit(X_scaled, y)

    return best_clf, scaler, best_name


def save_model(clf, scaler, model_name):
    """Save the model and scaler for inference.

    Args:
        clf: (estimator) trained classifier.
        scaler: (StandardScaler) fitted scaler.
        model_name: (str) classifier name, for the log message.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "emg_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "emg_scaler.pkl")
    print(f"\n  Model saved: models/emg_model.pkl ({model_name})")
    print(f"  Scaler saved: models/emg_scaler.pkl")


def main():
    print("=" * 50)
    print("  KINEA - ML model training")
    print("=" * 50)

    # Use the CSV passed as argument, else the newest one in data/
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csvs = sorted(glob.glob(str(DATA_DIR / "emg_data_*.csv")))
        if not csvs:
            print("  No emg_data_*.csv found in data/")
            print("  Usage: python3 train_model.py emg_data_XXXXX.csv")
            return
        csv_path = csvs[-1]
        print(f"  Using newest CSV: {csv_path}")

    X, y, _ = load_data(csv_path)

    # Need a minimum amount of data to train a useful model
    if len(X) < 20:
        print("  Not enough data. Record more repetitions.")
        return

    clf, scaler, model_name = train_and_compare(X, y)
    save_model(clf, scaler, model_name)

    print("\n" + "=" * 50)
    print("  Done. Run python3 app.py to use the model.")
    print("=" * 50)


if __name__ == "__main__":
    main()
