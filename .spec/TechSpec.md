# FaceVitals: Technical Specification (TechSpec)
## Dual Runtime: Python 3.10+ / PyTorch & MathWorks MATLAB R2024b/R2025a

### 1. Technology Stack & Runtime Dependencies

#### Python Stack (Algorithmic Core, Simulation & Evaluation)
* **Language:** Python 3.10+
* **Numerical & Signal Processing:** `numpy` (>= 1.24), `scipy` (>= 1.10) for causal SOS Butterworth filters and Welch PSD.
* **Computer Vision & Tracking:** `opencv-python` (`cv2`) for Haar cascade face detection, optical flow, and CIE $L^*a^*b^*$ color conversion.
* **Deep Learning Framework:** `torch` (>= 2.0) for Causal TCN and State-Space Model architectures.
* **MATLAB File I/O:** `scipy.io` (`savemat`, `loadmat`) for native MATLAB v5/v7 compatibility and HDF5 for large matrices.

#### MATLAB Stack (MathWorks Evaluation & Edge Deployment)
* **Environment:** MATLAB R2024b / R2025a
* **Toolboxes Required:** 
  - Image Processing Toolbox (`imnoise`, `rgb2lab`, `imadjust`, `imfilter`, `imresize`)
  - Computer Vision Toolbox (`faceDetector`, `vision.PointTracker`, `assignDetectionsToTracks`)
  - Signal Processing Toolbox (`designfilt`, `butter`, `sosfilt`, `pwelch`)
  - MATLAB Coder & Raspberry Pi Blockset for standalone ARM64 C++ compilation.

---

### 2. Edge Deployment Budget (Raspberry Pi 5 Reference Target)
* **Target Hardware:** Broadcom BCM2712 Quad Cortex-A76 @ 2.4 GHz, 4GB/8GB LPDDR4X.
* **Peak Latency Budget:** $\le 30.0\,\text{ms/frame}$ (sustaining $\ge 30\,\text{fps}$ real-time throughput).
* **Peak Memory Footprint:** $\le 120\,\text{MB}$ RAM during continuous streaming.
* **Quantization & Math:** INT8 / FP16 quantized causal state-space convolutions with ARM NEON SIMD vectorization.

---

### 3. Core Module Specifications

```text
src/
├── preprocessing/
│   ├── face_tracker.py       # Relative-fraction face & 3-ROI bounding tracking
│   ├── retinex.py            # Multi-Scale Retinex illumination decomposition (C1)
│   ├── ita.py                # CIE L*a*b* Individual Typology Angle stratification (C2)
│   ├── occlusion.py          # Surgical mask detection & dynamic ROI reweighting (C7)
│   └── degradation.py        # Poisson-Gaussian, bicubic downsampling, 10 fps decimation
├── classical/
│   ├── rppg_pos.py           # Plane-Orthogonal-to-Skin (POS) projection
│   ├── rppg_chrom.py         # Chrominance (CHROM) projection
│   ├── filter_bank.py        # Causal Second-Order Sections (SOS) bandpass filter
│   └── harmonic_estimator.py # Harmonic comb peak tracking & parabolic interpolation
├── models/
│   ├── trace_student.py      # Causal TCN trace-space model (trainable on cluster)
│   └── biophysical_optics.py # Kubelka-Munk SpO2 & Windkessel BP hemodynamic models
├── evaluation/
│   ├── metrics.py            # MAE, RMSE, Pearson r, Bland-Altman cluster bootstrap
│   └── crlb.py               # Cramér-Rao Lower Bound feasibility calculator
└── submission/
    ├── mat_exporter.py       # Validated MATLAB struct exporter
    ├── process_round1.py     # Python Round 1 entrypoint
    └── process_round1.m      # Native MATLAB Round 1 entrypoint
```
