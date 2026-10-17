"""
Individual Typology Angle (ITA) Skin Stratification.
Objective quantification of epidermal pigmentation in CIE 1976 L*a*b* color space
(Constraint C2: Skin Tone Generalization & Fairness).
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any

class ITASkinClassifier:
    """
    Computes Individual Typology Angle (ITA) from specular-free, shadow-free skin ROI pixels.
    Maps to objective bins ITA-1 through ITA-6.
    """
    @staticmethod
    def compute_ita(roi_rgb: np.ndarray) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculates median ITA angle (in degrees) and stratum tag.
        """
        if roi_rgb.size == 0:
            return 0.0, "ITA-Undefined", {"L": 0.0, "b": 0.0}

        # Convert to CIE 1976 L*a*b*
        lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0] * (100.0 / 255.0)  # Scale to [0, 100]
        b = lab[:, :, 2] - 128.0            # Center b* [-128, 127]

        # Mask out extreme specular highlights (L > 95) and deep cast shadows (L < 15)
        valid = (L >= 15.0) & (L <= 95.0)
        if np.sum(valid) < 16:
            # Fallback if too few valid pixels
            valid = np.ones_like(L, dtype=bool)

        med_L = float(np.median(L[valid]))
        med_b = float(np.median(b[valid]))

        # Safeguard division by near-zero b*
        denom = med_b if abs(med_b) > 1e-4 else 1e-4
        ita_deg = float(np.arctan((med_L - 50.0) / denom) * (180.0 / np.pi))

        # Discretize into 6 objective strata
        if ita_deg > 55.0:
            stratum = "ITA-1"   # Very Light
        elif ita_deg > 41.0:
            stratum = "ITA-2"   # Light
        elif ita_deg > 28.0:
            stratum = "ITA-3"   # Intermediate
        elif ita_deg > 10.0:
            stratum = "ITA-4"   # Tan
        elif ita_deg > -30.0:
            stratum = "ITA-5"   # Brown
        else:
            stratum = "ITA-6"   # Dark

        return ita_deg, stratum, {"L": med_L, "b": med_b}
