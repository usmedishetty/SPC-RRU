"""
Harmonic Comb Maximum-Likelihood Frequency Estimator.
Accurate Heart Rate estimation from filtered rPPG signals with harmonic demodulation,
parabolic peak refinement, and post-filtering SNR computation.
"""

import numpy as np
from scipy.signal import welch
from typing import Tuple, Dict, Any

class HarmonicCombEstimator:
    """
    Physiological Heart Rate Estimator using Harmonic Comb Matching.
    Avoids octave-jumping errors (half-rate / double-rate aliasing).
    """
    def __init__(self, fps: float = 30.0, min_hr_bpm: float = 42.0, max_hr_bpm: float = 180.0):
        self.fps = fps
        self.min_hz = min_hr_bpm / 60.0
        self.max_hz = max_hr_bpm / 60.0

    def estimate_hr(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Estimates Heart Rate (BPM), refined frequency (Hz), in-band SNR (dB), and confidence.
        """
        N = len(signal)
        if N < int(self.fps * 2.0):
            # Under minimum length (less than 2 seconds): compute dynamic zero-crossing rate if available
            zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
            if len(zero_crossings) > 1:
                duration_sec = max(1e-3, float(N) / self.fps)
                freq_est = float(np.clip(len(zero_crossings) / (2.0 * duration_sec), self.min_hz, self.max_hz))
            else:
                freq_est = float((self.min_hz + self.max_hz) / 2.0)
            return {
                "hr_bpm": float(freq_est * 60.0),
                "freq_hz": freq_est,
                "snr_db": -10.0,
                "confidence": 0.1,
                "psd_freqs": np.array([]),
                "psd_power": np.array([])
            }

        # Zero-pad for higher nominal spectral grid resolution
        nfft = max(2048, int(2 ** np.ceil(np.log2(N * 4))))
        freqs, psd = welch(
            signal,
            fs=self.fps,
            nperseg=min(N, int(self.fps * 8.0)),
            nfft=nfft,
            scaling='density'
        )

        # Restrict to physiological search range
        in_band = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        band_freqs = freqs[in_band]
        band_psd = psd[in_band]

        if len(band_psd) == 0 or np.all(band_psd == 0):
            center_hz = float((self.min_hz + self.max_hz) / 2.0)
            return {
                "hr_bpm": float(center_hz * 60.0),
                "freq_hz": center_hz,
                "snr_db": -10.0,
                "confidence": 0.0,
                "psd_freqs": freqs,
                "psd_power": psd
            }

        # Harmonic comb scoring: score(f) = P(f) + 0.5 * P(2f) + 0.25 * P(3f)
        comb_scores = np.zeros_like(band_psd)
        for i, f0 in enumerate(band_freqs):
            score = band_psd[i]
            # 2nd harmonic
            idx_2f = np.argmin(np.abs(freqs - 2.0 * f0))
            if idx_2f < len(psd):
                score += 0.5 * psd[idx_2f]
            # 3rd harmonic
            idx_3f = np.argmin(np.abs(freqs - 3.0 * f0))
            if idx_3f < len(psd):
                score += 0.25 * psd[idx_3f]
            comb_scores[i] = score

        best_idx_in_band = int(np.argmax(comb_scores))
        peak_hz = band_freqs[best_idx_in_band]

        # Parabolic peak interpolation on log-PSD
        full_idx = np.where(freqs == peak_hz)[0][0]
        refined_hz = peak_hz
        if 0 < full_idx < len(psd) - 1:
            p_left = max(1e-12, psd[full_idx - 1])
            p_mid = max(1e-12, psd[full_idx])
            p_right = max(1e-12, psd[full_idx + 1])
            log_l = np.log(p_left)
            log_m = np.log(p_mid)
            log_r = np.log(p_right)
            denom = log_l - 2.0 * log_m + log_r
            if abs(denom) > 1e-6:
                delta = 0.5 * (log_l - log_r) / denom
                delta_f = freqs[1] - freqs[0]
                refined_hz = peak_hz + delta * delta_f

        # Calculate in-band SNR
        pulse_band = (freqs >= refined_hz - 0.15) & (freqs <= refined_hz + 0.15)
        pulse_power = float(np.sum(psd[pulse_band]))
        noise_power = float(np.sum(psd[in_band])) - pulse_power
        if noise_power > 0 and pulse_power > 0:
            snr_db = float(10.0 * np.log10(pulse_power / noise_power))
        else:
            snr_db = -10.0

        # Confidence based on SNR and peak prominence
        confidence = float(np.clip((snr_db + 5.0) / 15.0, 0.05, 0.99))
        hr_bpm = float(refined_hz * 60.0)

        return {
            "hr_bpm": hr_bpm,
            "freq_hz": float(refined_hz),
            "snr_db": snr_db,
            "confidence": confidence,
            "psd_freqs": freqs,
            "psd_power": psd
        }
