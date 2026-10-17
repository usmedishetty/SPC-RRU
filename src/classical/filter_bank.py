"""
Digital Filter Bank for Physiological Frequency Isolation.
Causal 4th-order Butterworth bandpass filter (0.7 - 3.0 Hz, 42 - 180 bpm)
implemented using Second-Order Sections (SOS) for numerical stability.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from typing import Tuple, Optional

class PhysiologicalFilterBank:
    """
    Causal IIR filter bank with streaming state preservation.
    """
    def __init__(self, fps: float = 30.0, low_hz: float = 0.7, high_hz: float = 3.0, order: int = 4):
        self.fps = fps
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

        nyquist = fps / 2.0
        # Ensure upper cutoff is strictly below Nyquist (essential for 10 fps, where Nyquist = 5 Hz)
        safe_high = min(self.high_hz, nyquist - 0.2)
        safe_low = min(self.low_hz, safe_high - 0.2)
        if safe_low <= 0:
            safe_low = 0.5

        # Design Second-Order Sections (SOS)
        self.sos = butter(
            self.order,
            [safe_low, safe_high],
            btype='bandpass',
            fs=self.fps,
            output='sos'
        )
        self.zi = sosfilt_zi(self.sos)

    def filter_batch(self, signal: np.ndarray) -> np.ndarray:
        """
        Filters an offline 1D signal batch with causal SOS filter.
        """
        if len(signal) < 8:
            return signal
        filtered, _ = sosfilt(self.sos, signal, zi=self.zi * signal[0])
        return filtered

    def filter_stream(self, sample: float) -> float:
        """
        Step-wise causal filter for real-time edge streaming.
        """
        out, self.zi = sosfilt(self.sos, [sample], zi=self.zi)
        return float(out[0])

    def reset(self):
        """Resets internal filter delay states."""
        self.zi = sosfilt_zi(self.sos)
