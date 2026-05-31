from gpiozero import MCP3008
import time
import numpy as np


class Reader:
    """Reads the EMG signal from an MCP3008 ADC and turns each sampling
    window into a small set of time-domain features."""

    def __init__(self, channel, vref, sample_rate, samples_per_window):
        """
        Args:
            channel: (int) MCP3008 channel where the EMG sensor is wired.
            vref: (float) ADC reference voltage, used to scale readings to volts.
            sample_rate: (int) target samples per second inside a window.
            samples_per_window: (int) number of samples used per feature window.
        """
        self.adc_reader = MCP3008(channel)
        self.vref = vref
        self.sample_rate = sample_rate
        self.samples_per_window = samples_per_window

    def read_voltage(self):
        """Read one sample and scale it to volts.

        Returns:
            voltage: (float) sensor voltage in the 0..vref range.
        """
        return self.adc_reader.value * self.vref

    def collect_window(self):
        """Collect a full window of samples at a fixed rate.

        Returns:
            samples: (list) raw voltage samples for one window.
        """
        samples = []
        interval = 1.0 / self.sample_rate
        for _ in range(self.samples_per_window):
            samples.append(self.read_voltage())
            time.sleep(interval)
        return samples

    def extract_features(self):
        """Compute five time-domain EMG features from one window.

        Feature order must match the training script: [RMS, MAV, ZC, WL, VAR].

        Returns:
            features: (list) [rms, mav, zc, wl, var] for the window.
        """
        w = np.array(self.collect_window())
        centered = w - np.mean(w)  # Remove DC offset before counting crossings
        return [
            np.sqrt(np.mean(w ** 2)),                 # RMS
            np.mean(np.abs(w)),                       # MAV, mean absolute value
            np.sum(np.diff(np.sign(centered)) != 0),  # ZC, zero crossings
            np.sum(np.abs(np.diff(w))),               # WL, waveform length
            np.var(w),                                # VAR, variance
        ]
