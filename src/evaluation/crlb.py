"""
Cramér-Rao Lower Bound (CRLB) Feasibility Calculator.
Derives the theoretical physical estimation limit for heart rate frequency
from optical SNR, observation window length T, and frame rate fs.
"""

import numpy as np
from typing import Dict, Any

class CRLBFeasibilityCalculator:
    """
    Computes Fisher Information and CRLB for rPPG cardiac frequency estimation.
    """
    @staticmethod
    def compute_bound(snr_db: float, window_sec: float, fps: float) -> Dict[str, Any]:
        """
        Calculates minimum theoretical standard deviation of HR estimation.
        Formula:
          Var(f) >= 12 / ((2*pi)^2 * SNR_lin * T * (N^2 - 1))
          sigma_HR_min = 60 * sqrt(Var(f)) bpm
        """
        T = max(0.5, float(window_sec))
        fs = max(1.0, float(fps))
        N = T * fs

        # Linear SNR from dB
        snr_lin = 10.0 ** (snr_db / 10.0)
        snr_lin = max(1e-4, snr_lin)

        # Fisher Information denominator term
        denom = (2.0 * np.pi) ** 2 * snr_lin * T * (N ** 2 - 1.0)
        if denom <= 0:
            var_f = 1e6
        else:
            var_f = 12.0 / denom

        sigma_f = np.sqrt(var_f)
        crlb_hr_bpm = float(60.0 * sigma_f)

        # Infeasibility criteria:
        # 1. Linear SNR is below critical threshold (-15 dB)
        # 2. Window is below 2.5 seconds
        # 3. Bound exceeds 15.0 bpm (resolution collapse)
        is_infeasible = (snr_db < -15.0) or (T < 2.5) or (crlb_hr_bpm > 15.0)

        return {
            "crlb_hr_bpm": crlb_hr_bpm,
            "crlb_freq_hz": float(sigma_f),
            "snr_linear": float(snr_lin),
            "is_physically_infeasible": is_infeasible,
            "window_sec": T,
            "fps": fs
        }
