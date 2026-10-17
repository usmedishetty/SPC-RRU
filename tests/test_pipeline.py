"""
Comprehensive Unit & Integration Test Suite for FaceVitals Pipeline.
Tests algorithmic correctness, numerical stability, and schema compliance
without any hardcoded mock data.
"""

import os
import sys
import numpy as np
import pytest
import torch

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.face_tracker import FacialROITracker
from src.preprocessing.retinex import RetinexDecomposer
from src.preprocessing.ita import ITASkinClassifier
from src.preprocessing.occlusion import MaskOcclusionDetector
from src.preprocessing.degradation import PhysicalDegradationSimulator
from src.classical.filter_bank import PhysiologicalFilterBank
from src.classical.rppg_pos import POSAlgorithm
from src.classical.rppg_chrom import CHROMAlgorithm
from src.classical.harmonic_estimator import HarmonicCombEstimator
from src.evaluation.metrics import ClinicalMetricsEngine
from src.evaluation.crlb import CRLBFeasibilityCalculator
from src.models.trace_student import TraceSpaceStudent
from src.models.biophysical_optics import KubelkaMunkSkinOptics, WindkesselBPModel
from src.submission.mat_exporter import SubmissionMATExporter

def test_face_tracker():
    tracker = FacialROITracker()
    # Synthetic frame 240p
    frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
    res = tracker.track_and_extract(frame)
    assert "face_box" in res
    assert "rois" in res
    assert res["rois"]["forehead"].shape[0] > 0
    assert res["rois"]["left_cheek"].shape[0] > 0
    assert res["rois"]["right_cheek"].shape[0] > 0
    assert "mean_rgb" in res

def test_retinex():
    decomposer = RetinexDecomposer(scales=[15, 80])
    frame = np.random.randint(10, 40, (120, 160, 3), dtype=np.uint8)
    out = decomposer.multi_scale_retinex(frame)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8

def test_ita_classification():
    # Light skin ROI (high L*, low b*)
    light_roi = np.full((30, 30, 3), [220, 180, 160], dtype=np.uint8)
    ita_deg, stratum, _ = ITASkinClassifier.compute_ita(light_roi)
    assert isinstance(ita_deg, float)
    assert stratum.startswith("ITA-")

    # Dark skin ROI (low L*)
    dark_roi = np.full((30, 30, 3), [60, 45, 35], dtype=np.uint8)
    ita_dark, stratum_dark, _ = ITASkinClassifier.compute_ita(dark_roi)
    assert ita_dark < ita_deg

def test_occlusion_detector():
    # Unmasked: forehead and cheeks identical skin color
    fh = np.full((20, 20, 3), [200, 150, 130], dtype=np.uint8)
    lc = np.full((20, 20, 3), [200, 150, 130], dtype=np.uint8)
    rc = np.full((20, 20, 3), [200, 150, 130], dtype=np.uint8)
    is_masked, _, weights = MaskOcclusionDetector.detect_occlusion(fh, lc, rc)
    assert not is_masked
    assert weights["forehead"] == 0.50

    # Masked: cheeks covered with blue surgical mask [R=50, G=150, B=220]
    lc_mask = np.full((20, 20, 3), [50, 150, 220], dtype=np.uint8)
    rc_mask = np.full((20, 20, 3), [50, 150, 220], dtype=np.uint8)
    is_masked_m, _, weights_m = MaskOcclusionDetector.detect_occlusion(fh, lc_mask, rc_mask)
    assert is_masked_m
    assert weights_m["forehead"] == 1.0

def test_physical_degradation():
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    # Low light C1
    dim = PhysicalDegradationSimulator.simulate_low_light(frame, target_lux=30.0, auto_exposure=True)
    assert dim.shape == frame.shape
    assert dim.dtype == np.uint8

    # Low resolution C4
    down = PhysicalDegradationSimulator.downsample_resolution(frame, (80, 60))
    assert down.shape == (60, 80, 3)

    # Frame decimation C5
    seq = [frame.copy() for _ in range(30)]
    dec, target_fps = PhysicalDegradationSimulator.decimate_temporal(seq, 30.0, 10.0, mode="boxcar")
    assert len(dec) == 10
    assert target_fps == 10.0

def test_filter_bank():
    fb = PhysiologicalFilterBank(fps=30.0, low_hz=0.7, high_hz=3.0)
    # Generate 1.2 Hz pulse (72 bpm) + 10 Hz noise
    t = np.linspace(0, 10, 300)
    sig = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 10.0 * t)
    filtered = fb.filter_batch(sig)
    assert len(filtered) == len(sig)
    # 10 Hz should be attenuated
    assert np.std(filtered) < np.std(sig)

def test_rppg_pos_and_chrom():
    # Synthesize 3-channel RGB trace with subtle cardiac modulation at 1.3 Hz (78 bpm)
    t = np.linspace(0, 10, 300)
    cardiac = 0.02 * np.sin(2 * np.pi * 1.3 * t)
    rgb_traces = np.zeros((300, 3))
    rgb_traces[:, 0] = 180.0 + cardiac * 180.0 # Red
    rgb_traces[:, 1] = 140.0 + cardiac * 2.0 * 140.0 # Green (higher absorption)
    rgb_traces[:, 2] = 110.0 + cardiac * 0.5 * 110.0 # Blue

    pos = POSAlgorithm(fps=30.0, win_sec=1.6)
    bvp_pos = pos.extract_bvp(rgb_traces)
    assert len(bvp_pos) == 300

    chrom = CHROMAlgorithm(fps=30.0, win_sec=1.6)
    bvp_chrom = chrom.extract_bvp(rgb_traces)
    assert len(bvp_chrom) == 300

def test_harmonic_estimator():
    estimator = HarmonicCombEstimator(fps=30.0)
    # Clean 1.25 Hz pulse -> exactly 75 bpm
    t = np.linspace(0, 10, 300)
    sig = np.sin(2 * np.pi * 1.25 * t) + 0.3 * np.sin(2 * np.pi * 2.50 * t)
    res = estimator.estimate_hr(sig)
    assert abs(res["hr_bpm"] - 75.0) < 1.5 # Within 1.5 bpm
    assert res["snr_db"] > 0.0
    assert res["confidence"] > 0.5

def test_metrics_engine():
    y_true = np.array([70.0, 75.0, 80.0, 85.0])
    y_pred = np.array([71.0, 74.0, 81.0, 84.0])
    mae = ClinicalMetricsEngine.compute_mae(y_true, y_pred)
    assert abs(mae - 1.0) < 1e-4
    rmse = ClinicalMetricsEngine.compute_rmse(y_true, y_pred)
    assert abs(rmse - 1.0) < 1e-4
    r = ClinicalMetricsEngine.compute_pearson_r(y_true, y_pred)
    assert r > 0.95

    ba = ClinicalMetricsEngine.compute_bland_altman(y_true, y_pred)
    assert abs(ba["mean_bias"]) < 1e-4
    assert len(ba["loa_vector"]) == 2

def test_crlb_calculator():
    bound_clean = CRLBFeasibilityCalculator.compute_bound(snr_db=10.0, window_sec=8.0, fps=30.0)
    assert not bound_clean["is_physically_infeasible"]
    assert bound_clean["crlb_hr_bpm"] < 2.0

    bound_noisy = CRLBFeasibilityCalculator.compute_bound(snr_db=-20.0, window_sec=2.0, fps=30.0)
    assert bound_noisy["is_physically_infeasible"]

def test_biophysical_optics():
    optics = KubelkaMunkSkinOptics()
    r_norm = optics.forward_reflectance(c_mel=0.3, v_blood=0.03, spo2=0.98)
    assert len(r_norm) == 3
    assert np.all(r_norm >= 0.0) and np.all(r_norm <= 1.0)

    # Test Windkessel BP
    t = np.linspace(0, 5, 150)
    bvp = np.sin(2 * np.pi * 1.2 * t)
    feats = WindkesselBPModel.extract_sdppg_features(bvp, fps=30.0)
    assert "agi" in feats
    sbp, dbp = WindkesselBPModel.estimate_blood_pressure(feats)
    assert 90.0 <= sbp <= 160.0
    assert 50.0 <= dbp <= 100.0

def test_trace_student_model():
    model = TraceSpaceStudent(num_patches=16, metadata_dim=5, latent_dim=64)
    # Batch size 2, 48 channels (16*3), 150 frames (5 seconds @ 30fps)
    x = torch.randn(2, 48, 150)
    meta = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.5], [0.1, 0.33, 0.1, 1.0, -0.5]])
    out = model(x, meta)
    assert out["waveform"].shape == (2, 150)
    assert out["latent_z"].shape == (2, 64)
    assert "hr" in out
    assert "spo2" in out
    assert "sbp" in out
    assert "dbp" in out

def test_mat_export_and_validation(tmp_path):
    out_file = str(tmp_path / "Round1_Baseline_None.mat")
    video = np.random.randint(0, 255, (30, 60, 80, 3), dtype=np.uint8)
    roi_data = np.zeros((30, 3, 4), dtype=np.int32)
    ref = np.full(30, 72.0)
    proc = np.full(30, 71.8)
    metrics = {"MAE": 0.2, "RMSE": 0.2, "Pearson_r": 0.99, "BlandAltman_LoA": [-0.5, 0.5]}
    params = {
        "camera_resolution": "80x60",
        "frame_rate": 30.0,
        "illuminance_lux": 300.0,
        "camera_to_subject_distance": "50cm",
        "subject_fitzpatrick_skin_type": "TypeIII",
        "subject_age": 25,
        "constraint_applied": "None"
    }

    SubmissionMATExporter.export_mat(out_file, video, roi_data, ref, proc, metrics, params)
    val = SubmissionMATExporter.validate_mat(out_file)
    assert val["valid"] is True
    assert val["num_frames"] == 30
