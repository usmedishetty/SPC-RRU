"""
Plane-Orthogonal-to-Skin (POS) rPPG Projection Algorithm.
Wang et al., IEEE TBME 2017: Algorithmic extraction of pulsatile blood volume
from temporal RGB signals without training.
"""

import numpy as np
from typing import Tuple

class POSAlgorithm:
    """
    Implements overlap-add Plane-Orthogonal-to-Skin (POS) algorithm
    across sliding temporal windows.
    """
    def __init__(self, fps: float = 30.0, win_sec: float = 1.6):
        self.fps = fps
        self.win_len = max(8, int(fps * win_sec))

    def extract_bvp(self, rgb_traces: np.ndarray) -> np.ndarray:
        """
        Extracts 1D Blood Volume Pulse (BVP) from RGB traces.
        Args:
          rgb_traces: [N x 3] array of temporal mean RGB values.
        Returns:
          bvp: [N] extracted continuous pulse waveform.
        """
        N = len(rgb_traces)
        if N < self.win_len:
            # Fallback for ultra-short windows
            C = rgb_traces.T # [3 x N]
            mean_c = np.mean(C, axis=1, keepdims=True) + 1e-6
            Cn = C / mean_c - 1.0
            S1 = Cn[0, :] - Cn[1, :]
            S2 = Cn[0, :] + Cn[1, :] - 2.0 * Cn[2, :]
            std2 = np.std(S2)
            alpha = np.std(S1) / (std2 if std2 > 1e-6 else 1e-6)
            return S1 + alpha * S2

        # Standard overlap-add sliding window POS
        H = np.zeros(N, dtype=np.float64)
        weights = np.zeros(N, dtype=np.float64)

        for m in range(0, N - self.win_len + 1):
            sub_c = rgb_traces[m : m + self.win_len, :].T # [3 x win_len]
            mean_c = np.mean(sub_c, axis=1, keepdims=True) + 1e-6
            Cn = sub_c / mean_c - 1.0

            S1 = Cn[0, :] - Cn[1, :]
            S2 = Cn[0, :] + Cn[1, :] - 2.0 * Cn[2, :]
            std2 = np.std(S2)
            alpha = np.std(S1) / (std2 if std2 > 1e-6 else 1e-6)
            h_win = S1 + alpha * S2

            # Zero-mean center the window
            h_win -= np.mean(h_win)

            H[m : m + self.win_len] += h_win
            weights[m : m + self.win_len] += 1.0

        # Normalize overlap weights
        valid_idx = weights > 0
        H[valid_idx] /= weights[valid_idx]

        # Handle boundary extrapolation if needed
        if not np.all(valid_idx):
            H[~valid_idx] = 0.0

        return H
