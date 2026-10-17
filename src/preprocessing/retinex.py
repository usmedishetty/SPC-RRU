"""
Multi-Scale Retinex (MSR) Illumination Decomposition.
Decouples high-frequency tissue reflectance from low-frequency ambient illuminance
under degraded lighting conditions (Constraint C1, illuminance <= 50 lux).
"""

import cv2
import numpy as np
from typing import List

class RetinexDecomposer:
    """
    Multi-Scale Retinex algorithm for robust low-light feature enhancement.
    Used strictly for face detection and tracking under severe low-light (<= 50 lux).
    """
    def __init__(self, scales: List[int] = None):
        if scales is None:
            self.scales = [15, 80, 250]
        else:
            self.scales = scales

    def single_scale_retinex(self, img_channel: np.ndarray, sigma: float) -> np.ndarray:
        """
        SSR: R_i(x, y) = log(I_i(x, y)) - log(I_i(x, y) * G(x, y; sigma))
        """
        img_flt = np.maximum(img_channel.astype(np.float32), 1.0)
        blur = cv2.GaussianBlur(img_flt, (0, 0), sigma)
        blur = np.maximum(blur, 1.0)
        retinex = np.log(img_flt) - np.log(blur)
        return retinex

    def multi_scale_retinex(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        MSR combines multiple Gaussian surround scales to preserve both dynamic range
        and local color tonal rendition.
        """
        h, w, c = img_rgb.shape
        msr = np.zeros((h, w, c), dtype=np.float32)
        weight = 1.0 / len(self.scales)

        for channel in range(c):
            ch_data = img_rgb[:, :, channel]
            for s in self.scales:
                msr[:, :, channel] += weight * self.single_scale_retinex(ch_data, s)

        # Normalize across dynamic range
        out_rgb = np.zeros_like(img_rgb)
        for channel in range(c):
            ch_res = msr[:, :, channel]
            min_val = np.percentile(ch_res, 1)
            max_val = np.percentile(ch_res, 99)
            if max_val > min_val:
                norm = np.clip((ch_res - min_val) / (max_val - min_val), 0.0, 1.0)
            else:
                norm = np.zeros_like(ch_res)
            out_rgb[:, :, channel] = np.uint8(norm * 255.0)

        return out_rgb
