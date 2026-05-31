import joblib
import numpy as np


class Predictor:
    """Loads the trained EMG model and turns feature windows into a
    stable open/closed decision."""

    def __init__(self, model, scaler, stability):
        """
        Args:
            model: (str|Path) path to the saved classifier (.pkl).
            scaler: (str|Path) path to the saved StandardScaler (.pkl).
            stability: (int) number of equal predictions needed to act.
        """
        self.model = joblib.load(model)
        self.scaler = joblib.load(scaler)
        self.stability = stability
        self.last_predictions = []

    def predict(self, features):
        """Scale one feature window and predict the gesture.

        Args:
            features: (list) [rms, mav, zc, wl, var] for one window.

        Returns:
            prediction: (int) 0 for closed, 1 for open.
        """
        scaled_features = self.scaler.transform([features])
        prediction = self.model.predict(scaled_features)[0]

        # Keep only the last `stability` predictions
        self.last_predictions.append(prediction)
        if len(self.last_predictions) > self.stability:
            self.last_predictions.pop(0)
        return prediction

    def stable_prediction(self):
        """Return a decision only when the recent predictions all agree.

        Returns:
            label: (int|None) 0 or 1 if stable, None otherwise.
        """
        if len(self.last_predictions) < self.stability:
            return None
        if all(p == self.last_predictions[0] for p in self.last_predictions):
            return self.last_predictions[0]
        return None
