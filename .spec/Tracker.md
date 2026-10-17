# FaceVitals: Implementation Tracker & Milestone Matrix

### 1. Sprint 1 Milestones (18 Oct – 25 Oct 2026)
- [x] Syzygy SDD Specification suite (`.spec/` 8 documents) created.
- [ ] Relative-fraction Face and Multi-ROI Tracker (`src/preprocessing/face_tracker.py`).
- [ ] Multi-Scale Retinex low-light illuminance decomposer (`src/preprocessing/retinex.py`).
- [ ] CIE $L^*a^*b^*$ ITA skin-tone classifier (`src/preprocessing/ita.py`).
- [ ] Surgical mask detector and dynamic ROI re-weighting (`src/preprocessing/occlusion.py`).
- [ ] Physical degradation simulator for C1, C4, C5 (`src/preprocessing/degradation.py`).
- [ ] Classical POS and CHROM rPPG projection engines (`src/classical/`).
- [ ] Causal SOS Butterworth bandpass filter & Harmonic Comb Peak Estimator (`src/classical/`).
- [ ] Statistical Evaluation Engine & Cramér-Rao Bound calculator (`src/evaluation/`).
- [ ] Production MATLAB `.mat` Exporter & submission validator (`src/submission/`).
- [ ] Native MATLAB Round 1 pipeline script (`process_round1.m`).
- [ ] Pytest verification suite across all core modules.
- [ ] Official SP Cup team registration completed on IEEE portal.

---

### 2. Sprint 2 Milestones (26 Oct – 15 Nov 2026)
- [ ] Ingestion and indexing of BUAA-MIHR, MMPD, and UBFC-rPPG datasets.
- [ ] Simulator calibration against the BUAA-MIHR 11-step physical lux ladder.
- [ ] High-compute synthetic pulse-injection engine generation on cluster.
- [ ] Trace-Space Causal Student training and model weight export.
- [ ] Seven-constraint evaluation suite runs (C1 through C7).

---

### 3. Sprint 3 Milestones (16 Nov – 30 Nov 2026: Round 1 Submission)
- [ ] Generate submission containers: `Round1_Baseline_None.mat`, `Round1_LowLight_C1.mat`, etc.
- [ ] Verify `.mat` loadability directly in MATLAB R2024b/R2025a.
- [ ] Draft 4-page IEEE conference format technical report with Bland-Altman and CRLB figures.
- [ ] Upload final Round 1 package to IEEE SPS portal.
