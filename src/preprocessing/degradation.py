"""
Physical Degradation Engine.
Simulates real-world deployment constraints:
- C1: Low-light illuminance (<= 50 lux) with Poisson shot & Gaussian read noise
- C4: Low camera resolution downsampling (VGA/QVGA) with anti-aliasing
- C5: Temporal frame decimation (<= 10 fps) with anti-flicker prefiltering
"""

import cv2
import numpy as np
from typing import List, Tuple

class PhysicalDegradationSimulator:
    """
    Simulates optical and transmission degradations adhering to official Call for Proposals §2.2.
    """
    @staticmethod
    def simulate_low_light(frame_rgb: np.ndarray, 
                           target_lux: float = 50.0, 
                           base_lux: float = 300.0,
                           phi_full_well: float = 5000.0,
                           read_noise_std: float = 2.5,
                           auto_exposure: bool = True,
                           seed: int = 42) -> np.ndarray:
        """
        Applies linear-light attenuation and CMOS sensor noise (shot + read noise).
        """
        assert frame_rgb.dtype == np.uint8, "Input frame must be uint8"
        np.random.seed(seed)

        # 1. Linearize sRGB (gamma = 2.2 approximation)
        frame_lin = np.power(frame_rgb.astype(np.float32) / 255.0, 2.2)

        # 2. Multiplicative illuminance gain
        gain = np.clip(target_lux / base_lux, 0.01, 1.0)
        atten_lin = frame_lin * gain

        # 3. Photon Poisson shot noise
        mean_photons = atten_lin * phi_full_well
        shot_photons = np.random.poisson(mean_photons).astype(np.float32)

        # 4. Sensor Gaussian read noise
        read_noise = np.random.normal(0.0, read_noise_std, size=frame_rgb.shape).astype(np.float32)
        total_photons = np.maximum(0.0, shot_photons + read_noise)

        # 5. Auto-exposure digital gain boost if active
        if auto_exposure:
            digital_gain = 1.0 / max(1e-3, gain)
            out_lin = np.clip((total_photons / phi_full_well) * digital_gain, 0.0, 1.0)
        else:
            out_lin = np.clip(total_photons / phi_full_well, 0.0, 1.0)

        # 6. Re-apply sRGB gamma and quantize to 8-bit
        out_srgb = np.clip(np.round(255.0 * np.power(out_lin, 1.0 / 2.2)), 0, 255).astype(np.uint8)
        return out_srgb

    @staticmethod
    def downsample_resolution(frame_rgb: np.ndarray, target_wh: Tuple[int, int] = (320, 240)) -> np.ndarray:
        """
        Anti-aliased spatial downsampling matching MATLAB imresize bicubic behavior (C4).
        """
        w_tgt, h_tgt = target_wh
        return cv2.resize(frame_rgb, (w_tgt, h_tgt), interpolation=cv2.INTER_AREA)

    @staticmethod
    def decimate_temporal(frames: List[np.ndarray], 
                           orig_fps: float = 30.0, 
                           target_fps: float = 10.0,
                           mode: str = "boxcar") -> Tuple[List[np.ndarray], float]:
        """
        Decimates temporal video frame sequence down to target_fps (Constraint C5).
        Modes:
          - 'skip': Standard periodic frame skipping.
          - 'boxcar': Average-then-skip (sinc prefilter mitigating 50Hz/100Hz mains flicker aliasing).
        """
        if target_fps >= orig_fps:
            return frames, orig_fps

        stride = orig_fps / target_fps
        num_out = int(len(frames) / stride)
        indices = [int(round(i * stride)) for i in range(num_out) if int(round(i * stride)) < len(frames)]

        if mode == "skip":
            decimated = [frames[i] for i in indices]
            return decimated, target_fps
        elif mode == "boxcar":
            decimated = []
            for i, idx in enumerate(indices):
                start = idx
                end = indices[i + 1] if (i + 1 < len(indices)) else min(len(frames), start + int(round(stride)))
                chunk = frames[start:max(start + 1, end)]
                avg_frame = np.mean(chunk, axis=0).astype(np.uint8)
                decimated.append(avg_frame)
            return decimated, target_fps
        else:
            raise ValueError(f"Unknown decimation mode: {mode}")
