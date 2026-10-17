"""
Mask Occlusion Detector & Dynamic ROI Reweight Engine.
Identifies lower-face occlusion (Constraint C7: >= 40% facial coverage, surgical/cloth masks)
and dynamically shifts extraction weight to exposed forehead/periocular regions.
"""

import cv2
import numpy as np
from typing import Dict, Tuple

class MaskOcclusionDetector:
    """
    Detects lower-face occlusion using color-space skin probability discrepancy
    between upper face (forehead) and lower face (cheeks/chin).
    """
    @staticmethod
    def detect_occlusion(forehead_crop: np.ndarray, 
                         left_cheek_crop: np.ndarray, 
                         right_cheek_crop: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """
        Determines if lower-face is masked and outputs optimal fusion weights.
        Returns:
          is_masked: bool
          mask_confidence: float [0, 1]
          weights: dict of float weights for {'forehead', 'left_cheek', 'right_cheek'}
        """
        if forehead_crop.size == 0 or left_cheek_crop.size == 0 or right_cheek_crop.size == 0:
            return False, 0.0, {"forehead": 0.50, "left_cheek": 0.25, "right_cheek": 0.25}

        # Convert to YCrCb for robust skin color thresholding
        fh_ycrcb = cv2.cvtColor(forehead_crop, cv2.COLOR_RGB2YCrCb)
        lc_ycrcb = cv2.cvtColor(left_cheek_crop, cv2.COLOR_RGB2YCrCb)
        rc_ycrcb = cv2.cvtColor(right_cheek_crop, cv2.COLOR_RGB2YCrCb)

        # Standard human skin locus in Cr-Cb space
        def skin_fraction(crop_ycrcb: np.ndarray) -> float:
            cr = crop_ycrcb[:, :, 1]
            cb = crop_ycrcb[:, :, 2]
            # General skin color mask: 133 <= Cr <= 173 and 77 <= Cb <= 127
            mask = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130)
            return float(np.mean(mask))

        fh_skin = skin_fraction(fh_ycrcb)
        lc_skin = skin_fraction(lc_ycrcb)
        rc_skin = skin_fraction(rc_ycrcb)
        cheek_skin = (lc_skin + rc_skin) / 2.0

        # Masked condition: Forehead has skin characteristics, but cheeks deviate sharply (cloth/surgical mask)
        # or have distinct non-skin color (e.g. blue/white/black surgical mask)
        diff = max(0.0, fh_skin - cheek_skin)
        is_masked = (diff > 0.35) or (cheek_skin < 0.20 and fh_skin > 0.40)
        mask_confidence = float(np.clip(diff * 2.0, 0.0, 1.0))

        if is_masked:
            # Shift 100% weight to forehead when masked
            weights = {"forehead": 1.0, "left_cheek": 0.0, "right_cheek": 0.0}
        else:
            # Balanced anatomical weighting: 50% forehead, 25% each cheek
            weights = {"forehead": 0.50, "left_cheek": 0.25, "right_cheek": 0.25}

        return is_masked, mask_confidence, weights
