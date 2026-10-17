"""
Chrominance-based rPPG (CHROM) Projection Algorithm.
de Haan & Jeanne, IEEE TBME 2013: Robust pulse rate from chrominance features.
"""

import numpy as np

class CHROMAlgorithm:
    """
    Implements overlap-add Chrominance-based (CHROM) projection
    across sliding temporal windows.
    """
    def __init__(self, fps: float = 30.0, win_sec: float = 1.6):
        self.fps = fps
        self.win_len = max(8, int(fps * win_sec))

    def extract_bvp(self, rgb_traces: np.ndarray) -> np.ndarray:
        """
        Extracts 1D Blood Volume Pulse (BVP) from RGB traces via CHROM.
        """
        N = len(rgb_traces)
        if N < self.win_len:
            C = rgb_traces.T
            mean_c = np.mean(C, axis=1, keepdims=True) + 1e-6
            Cn = C / mean_c
            Xs = 3.0 * Cn[0, :] - 2.0 * Cn[1, :]
            Ys = 1.5 * Cn[0, :] + Cn[1, :] - 1.5 * Cn[2, :]
            std_y = np.std(Ys)
            alpha = np.std(Xs) / (std_y if std_y > 1e-6 else 1e-6)
            return Xs - alpha * Ys

        H = np.zeros(N, dtype=np.float64)
        weights = np.zeros(N, dtype=np.float64)

        for m in range(0, N - self.win_len + 1):
            sub_c = rgb_traces[m : m + self.win_len, :].T
            mean_c = np.mean(sub_c, axis=1, keepdims=True) + 1e-6
            Cn = sub_c / mean_c

            Xs = 3.0 * Cn[0, :] - 2.0 * Cn[1, :]
            Ys = 1.5 * Cn[0, :] + Cn[1, :] - 1.5 * Cn[2, :]
            std_y = np.std(Ys)
            alpha = np.std(Xs) / (std_y if std_y > 1e-6 else 1e-6)
            h_win = Xs - alpha * Ys

            h_win -= np.mean(h_win)
            H[m : m + self.win_len] += h_win
            weights[m : m + self.win_len] += 1.0

        valid_idx = weights > 0
        H[valid_idx] /= weights[valid_idx]
        if not np.all(valid_idx):
            H[~valid_idx] = 0.0

        return H
