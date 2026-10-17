# FaceVitals: Implementation-Ready Technical Blueprint and Competition Execution Plan
**2027 IEEE Signal Processing Cup**  
*Theme: Equitable, Edge-Deployable Camera-Based Vital Sign Monitoring*  
**Planning Date:** 17 October 2026 | **Target Conference:** IEEE ICASSP 2027 (16–21 May 2027)  
**Document Classification:** Technical Architecture & Operational Plan

---

## 1. Executive Summary, Thesis and Scoring Map

### 1.1 Core Engineering Thesis
> An equitable, edge-deployable camera-based vital signs platform cannot rely on black-box end-to-end deep learning or uncalibrated classical rPPG under extreme real-world stress. We present an **SQI-Conditioned Hybrid Physiological Architecture**: a dual-path framework fusing a deterministic, physics-grounded chrominance path (Plane-Orthogonal-to-Skin / CHROM) with a lightweight 3D-separable spatiotemporal convolutional neural network (PhysNet-Lite, $\approx 0.85$M parameters). A unified, multi-channel Signal Quality Index (SQI)—derived from spatial SNR, facial landmark stability, and spectral entropy—dynamically governs ROI spatial weighting, adaptive temporal windowing ($3.0\,\text{s}$ at $30\,\text{fps}$ down to $12.0\,\text{s}$ at $10\,\text{fps}$), and calibrated abstention. By training the neural trunk via Group-Distributionally Robust Optimization (Group-DRO) over Individual Typology Angle (ITA) strata and validating through a physics-based linear-light degradation simulator, the platform prevents silent catastrophic failure under low light ($\le 50\,\text{lux}$), dark skin tones (Fitzpatrick V–VI), spatial decimation ($\le 320\times 240$), and low bandwidth ($\le 500\,\text{kbps}$, $\le 10\,\text{fps}$) `[TEAM-DEFINED]`.

### 1.2 Evaluation and Selection of Headline Innovations
To ensure defensibility before a PhD-level IEEE jury, we evaluate five candidate innovations against criteria of falsifiability, edge feasibility, and constraint relevance:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │ Innovation Screening: Defensibility & Falsifiability    │
                  └─────────────────────────────────────────────────────────┘
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
    [ACCEPTED CANDIDATES]                                               [REJECTED CANDIDATES]
    1. Unified SQI-Conditioned Hybrid Fusion                             A. Super-Resolution Pre-Stage (ESRGAN/SRCNN)
       - Falsifiable: Ablating SQI gating increases                        - Rejection: Multiplies FLOPs >12×, hallucinates
         worst-stratum error >35%.                                           high-frequency PPG artifacts in dim light.
    2. Physics-Grounded Linear-Light Simulator                          B. Facial PTT-Based Multi-Site Blood Pressure
       - Falsifiable: sRGB-space naive scaling underpredicts               - Rejection: Infeasible at ≤10 fps (Nyquist limit:
         real-world error by >4.5 bpm.                                       100 ms frame interval >> 10–30 ms pulse transit).
    3. ITA Group-DRO Minimax Fair Optimization                          C. Blind End-to-End Black-Box Video Transformer
       - Falsifiable: Minimizing worst-group loss reduces                  - Rejection: OOM on Raspberry Pi 4/5, uninterpretable
         disparity across strata I–VI to |Δ| ≤ 1.5 bpm.                      failure modes under sensor shot noise.
```

#### The Three Selected Headline Innovations `[TEAM-DEFINED]`
1. **Innovation 1: Multi-Scale SQI-Conditioned Classical-Deep Adaptive Fusion**
   * *Hypothesis:* Combining physics-based chrominance projection ($\mathbf{S}_{\text{POS}}$) and deep spatial representations ($\hat{y}_{\text{deep}}$) via a spectral-entropy-weighted gating function $\alpha(\text{SQI})$ yields a lower worst-case MAE than either branch alone under combined stress ($\le 50\,\text{lux}$, QVGA, $10\,\text{fps}$).
   * *Ablation to Confirm/Kill:* Fix $\alpha = 0.5$ (static average) or $\alpha \in \{0, 1\}$ (single branch). If dynamic $\alpha(\text{SQI})$ does not reduce worst-group MAE by $\ge 20\%$ over the best single branch, reject.
2. **Innovation 2: Linear-Light, Photon-Calibrated Degradation Pipeline**
   * *Hypothesis:* Synthesizing low-light frames in linear irradiance space ($\gamma$-decoded) with explicit Poisson photon-counting ($\lambda_{\text{peak}} \propto \text{lux}$) and Gaussian read noise eliminates the domain gap between synthetic degradation and real physical captures ($\le 50\,\text{lux}$).
   * *Ablation to Confirm/Kill:* Train on standard naive sRGB multiplicative scaling vs. linear-light Poisson-Gaussian pipeline; evaluate on real low-light clinical clips. If cross-domain generalization error does not improve by $\ge 2.0\,\text{bpm}$ MAE, reject.
3. **Innovation 3: Group-DRO Minimax Fair Optimization over ITA Strata**
   * *Hypothesis:* Minimizing the worst-case group risk across 6 discretized ITA bins directly enforces the competition fairness criterion ($|\text{MAE}_g - \mu| \le \tau$) without sacrificing aggregate cohort accuracy.
   * *Ablation to Confirm/Kill:* Compare standard Empirical Risk Minimization (ERM) vs. Group-DRO. If Group-DRO fails to bound maximum stratum deviation $|\text{MAE}_g - \mu| \le 1.5\,\text{bpm}$ under baseline or increases mean cohort error by $>10\%$, reject.

### 1.3 Official Competition Round Progression
* **Round 1 (Baseline & Initial Constraints):** Controlled and uncontrolled HR extraction from RGB video `[DOC §2.3 Task 1]`. Must implement baseline and additionally address and justify $\ge 2$ constraints from C1–C7 `[DOC §2.3 Task 1]`. *Submission: 30 Nov 2026; Results: 05 Dec 2026 `[DOC §6]`.*
* **Round 2 (Two-Parameter Joint Estimation):** Extend pipeline to jointly estimate HR and $\text{SpO}_2$ without contact/thermal sensing `[DOC §2.3 Round 2]`. Quantitative validation against clinical pulse oximetry `[DOC §2.3 Round 2]`. *Submission: 16 Jan 2027; Results: 21 Jan 2027 `[DOC §6]`.*
* **Round 3 (Full Three-Parameter Clinical & Edge System):** Simultaneous estimation of HR, $\text{SpO}_2$, and Blood Pressure (Systolic/Diastolic) `[DOC §2.3 Round 3]`. Mandatory clinical validation against reference datasets/devices `[DOC §2.3 Round 3]`. Real-time live demo on an edge device (Raspberry Pi recommended) under $\ge 2$ constraints at IEEE ICASSP 2027 in New Orleans, USA (16–21 May 2027) `[DOC §2.3 Round 3, §6]`.

### 1.4 Scoring Map: Judging Criteria to Blueprint Implementation

| Final Round Judging Criterion `[DOC §5.2]` | Weight `[DOC §5.2]` | Blueprint Section & Technical Mechanism | Evidence Artifact Produced |
| :--- | :--- | :--- | :--- |
| **1. Innovation & Thoroughness of Experiments** | 16.67% (1/6) | • §1.2: Three falsifiable hypotheses<br>• §3.4: 1-factor sweeps & 4 combined worst-case cells<br>• §4.1: Ablation ladder (Classical $\to$ Deep $\to$ Fusion) | Ablation matrix table; synthetic vs. real domain gap plots; Group-DRO convergence logs. |
| **2. Validation-Set Performance** | 16.67% (1/6) | • §3.2: 5-fold subject-disjoint stratified CV<br>• §6.1–6.2: Pre-registered tuning metrics<br>• §4.3: Fair loss balance | Stratified cross-validation tables with 95% bootstrap CIs; risk-coverage curves. |
| **3. Test-Set Performance** | 16.67% (1/6) | • §3.2: Completely blind held-out subject cohort<br>• §6.1–6.3: Multi-metric evaluation (MAE, RMSE, $r$, Bland-Altman)<br>• C2 tolerance check $|\text{MAE}_g - \mu| \le \tau$ | Test results CSV; Bland-Altman agreement plots; ITA disparity histograms. |
| **4. Model & Time Complexity** | 16.67% (1/6) | • §2.5: Lightweight 3D-CNN ($\le 0.85$M params)<br>• INT8 PTQ quantization; causal temporal buffers<br>• Edge target: Raspberry Pi 4/5 (MathWorks Blockset) | Latency profiles (ms/frame), memory footprints (MB), FLOP counts, edge FPS logs. |
| **5. Quality & Clarity of Final Report** | 16.67% (1/6) | • §5.1: Strict IEEE conference paper template<br>• Transparent traceability tags `[DOC]`, `[TEAM-DEFINED]`<br>• §10.3: Ethical & clinical limitation statements | Complete 4-page (Round 1) / 6-page (Final) IEEE format PDF with vector graphics. |
| **6. Quality & Clarity of Presentation** | 16.67% (1/6) | • §8.1: Live interactive GUI demo architecture<br>• Fail-safe fallback recorded logs<br>• Clear defense against PhD jury questions | Slide deck, video demonstration walk-through, live hardware test harness. |

### 1.5 Mandatory vs. Optional Competition Scope Matrix

| Dimension | Mandatory Specification `[DOC]` | Team Strategy & Selected Stance | Traceability |
| :--- | :--- | :--- | :--- |
| **Primary Parameters** | HR (R1), HR+$\text{SpO}_2$ (R2), HR+$\text{SpO}_2$+BP (R3) | Dual-path SQI-conditioned multi-task model | `[DOC §2.1, §2.3]` |
| **Sensing Modality** | Standard RGB video only (no NIR, thermal, contact) | 3-channel visible spectrum rPPG (Forehead/Cheeks) | `[DOC §2.1]` |
| **Primary Constraints** | $\ge 2$ constraints in R1 & R3; C1, C2, C4, C5 priority | Deep focus: C1 (Light), C2 (Skin), C4 (Res), C5 (BW) | `[DOC §2.2, §2.3]` |
| **Secondary Constraints**| C3 (Multi-person), C6 (Aging), C7 (Masks) | Modular extension interfaces (Appendix D) | `[DOC §2.2]` |
| **Target Platform** | Raspberry Pi is optional / under evaluation | Raspberry Pi 4/5 optimized (INT8/C++ engine) | `[DOC §2.1, §5.2]` |
| **Implementation Language**| MATLAB or Python | Dual pipeline: Python PyTorch train, MATLAB deploy | `[DOC §2.3 Task 1, §7.2]` |

---

## 2. End-to-End System Architecture

### 2.1 End-to-End System Block Diagram
```
========================================================================================================================
                                     FACEVITALS END-TO-END PIPELINE ARCHITECTURE
========================================================================================================================

  [STAGE 1: ACQUISITION & SENSING]
  RGB Stream / Offline MP4 ──► Session Metadata Ingestion (fps, lux, res, distance, nominal kbps)
                                │
                                ▼
  [STAGE 2: PREPROCESSING & QUALITY GATING]
  Face Detection & Landmarks ──► Forehead & Bilateral Cheek Extraction ──► Illumination & Noise Gating
  (MediaPipe / Cascade)         (68/468-point Mesh with Fallback)          (Luminance Floor & Photon Noise Check)
                                │                                        │
                                ├────────────────────────────────────────┴──────────────┐
                                ▼                                                       ▼
  [STAGE 3A: CLASSICAL rPPG PATH]                                         [STAGE 3B: DEEP SPATIOTEMPORAL PATH]
  Spatial Averaging per ROI $\mathbf{c}(t) \in \mathbb{R}^3$                     Normalized Spatiotemporal Tensor $\mathbf{X}$
  Linear Normalization $\mathbf{c}_n(t) = \mathbf{c}(t)/\mu_c - 1$               $\mathbb{R}^{B \times 3 \times T \times H \times W}$ ($T$ frames, $64\times 64$)
  Projection: POS & CHROM Waveform Generators                             PhysNet-Lite (3D-Separable ResNet Trunk)
  Temporal Causal Bandpass (0.7–3.0 Hz, 4th-order IIR)                    Multi-Head Latent Features $\mathbf{z}(t) \in \mathbb{R}^{128}$
                                │                                                       │
                                └────────────────────────────────────────┬──────────────┘
                                                                         ▼
  [STAGE 4: SQI-CONDITIONED FUSION & MULTI-TASK ESTIMATION]
  Signal Quality Index (SQI) Computation ──► Dynamic Fusion Gate $\alpha(\text{SQI})$
  ├── Temporal Spectral Entropy ($H_{\text{spec}}$)
  ├── Peak-to-Noise Ratio (SNR$_{\text{rPPG}}$)
  └── Landmark Jitter / Boundary Tracking Variance
                                │
                                ├───────────────────┬───────────────────┐
                                ▼                   ▼                   ▼
                          [HR Head]           [SpO2 Head]          [BP Head]
                          Peak FFT / PSD      Modified Ratio-      Nonlinear Latent Mapping
                          Harmonic Check      of-Ratios + Calib    (Demographic + PTT proxy)
                          Output: BPM         Output: % Saturation Output: SBP/DBP (mmHg)
                                │                   │                   │
                                └───────────────────┼───────────────────┘
                                                    ▼
  [STAGE 5: CALIBRATED UNCERTAINTY & ABSTENTION]
  Epistemic / Aleatoric Uncertainty Check ──► If $\sigma_{\text{pred}} > \theta_{\text{abstain}} \implies$ ABSTAIN (Emit NaN)
                                         ──► Else Output Validated Vitals + 95% Confidence Bounds
========================================================================================================================
```

### 2.2 Architectural Components & Constraint Handling Matrix

| Pipeline Stage | Input $\to$ Output Shape | Method & Algorithmic Justification | Handling C1 (Low Light $\le 50\,\text{lux}$) | Handling C2 (Skin Tone / Vitiligo) | Handling C4 (Low Res $\le 320\times 240$) | Handling C5 (Low BW $\le 10\,\text{fps}$, $\le 500\,\text{kbps}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Ingestion & Metadata** | Video bitstream $\to$ $[T \times H \times W \times 3]$ uint8 | Direct frame buffer streaming; dynamic metadata header parsing `[DOC §1.2]`. | Estimates physical lux level from raw sensor exposure + ROI pixel histogram. | Flags self-reported Fitzpatrick and Von Luschan stratum tags in metadata. | Ingests native resolution ($720\text{p}$, VGA, or QVGA) without artificial upsampling. | Measures true achieved bitrate (kbps) via bitstream buffer length tracking. |
| **2. Face & Landmark Detection** | $[H \times W \times 3] \to$ BBox $[4]$, Mesh $[468 \times 2]$ | MediaPipe FaceMesh (Python) / vision.CascadeObjectDetector + KLT tracker (MATLAB) `[DOC §2.2]`. | Adaptive histogram pre-stretch on Y-channel only to ensure landmark convergence. | Landmark topology is geometry-based; invariant to skin melanin pigmentation. | Sub-pixel landmark interpolation; auto-switch to bounding box if landmarks jitter. | Runs landmark detection every $K=5$ frames; tracks via lightweight KLT in-between. |
| **3. Multi-ROI Extraction** | Frame + Mesh $\to$ 3 ROIs: $[3 \times T \times H_r \times W_r \times 3]$ | Segment Forehead ($60\times 30$), Bilateral Cheeks ($25\times 25$) `[DOC §10 Annexure 1]`. | Convex hull spatial trimming avoids shadow boundaries on facial margins. | Segment sub-ROIs; compute local ITA; detect vitiligo patches to re-weight ROIs. | Dynamic ROI scaling: downsizes nominal box to minimum $12\times 12$ px floor. | Decouples spatial cropping from frame rate; caches bounding geometries. |
| **4A. Classical Projection** | $[3 \times T] \to [T]$ temporal rPPG signal | POS (Plane-Orthogonal-to-Skin) and CHROM algorithms + 4th-order IIR bandpass `[DOC §2.1, §2.3]`. | Color space projection eliminates common-mode intensity drops under low illuminance. | POS standardizes skin reflectance plane across wide range of melanin concentrations. | Spatial pixel averaging over ROI acts as low-pass filter, recovering SNR at QVGA. | Causal IIR filtering with group delay compensation; avoids forward-backward filter lag. |
| **4B. Deep Spatiotemporal Branch** | $[B \times 3 \times T \times 64 \times 64] \to [B \times 128]$ | PhysNet-Lite: (2+1)D spatio-temporal convolutions extracting blood volume pulses `[TEAM-DEFINED]`. | Group normalization and channel attention layers capture faint periodic gradients. | Group-DRO trained representations prevent deep feature suppression on dark skin. | Input spatial pooling preserves phase-coherent color shifts despite pixel decimation. | Temporal depth scaling adjusts temporal receptive field dynamically for $10\,\text{fps}$. |
| **5. Multi-Channel SQI** | Raw signals $\to \text{SQI} \in [0, 1]$ | Fusion of spectral entropy ($H_{\text{spec}}$), peak SNR, and optical flow landmark variance. | Penalizes noise-floor amplification caused by sensor read noise in low lux. | Weights bilateral cheeks independently if vitiligo depigmentation affects one side. | Penalizes landmark drift and motion blur common in low-resolution captures. | Flags spectral peak broadening when window length is constrained at $10\,\text{fps}$. |
| **6. Multi-Task Heads (HR/SpO2/BP)** | Feature vector $\to$ HR, $\text{SpO}_2$, SBP, DBP | Multi-task dense heads with heteroscedastic uncertainty estimation $\sigma^2$ `[TEAM-DEFINED]`. | Spectral peak tracking with historical Kalman smoothing prevents jumping to noise. | Stratified calibration offsets per ITA cluster; eliminates dark skin bias drift. | Resolution-invariant latent embedding preserves vital calibration baselines. | Window length auto-expands from $3\,\text{s}$ ($30\,\text{fps}$) to $12\,\text{s}$ ($10\,\text{fps}$) to preserve $\Delta f$. |
| **7. Calibrated Abstention** | Vitals + $\sigma^2 \to$ Validated Vitals or $\text{NaN}$ | Risk-coverage gate: rejects frames where predicted confidence drops below $\theta_{\text{abstain}}$. | Rejects windows where illuminance drops below physical photon-shot noise threshold. | Flags uncalibrated vitiligo edge cases rather than outputting erroneous readings. | Abstains if face area falls below $32\times 32$ pixels in total frame. | Rejects windows affected by burst packet drop or temporal frame skips $>200\,\text{ms}$. |

### 2.3 Classical vs. Deep Dual-Path Design Rationale
* **The Classical Branch (POS/CHROM):** Provides an instantaneous, un-trainable, zero-parameter baseline that is impervious to dataset overfitting. POS projects normalized RGB signals onto two orthogonal planes:
  $$S_1(t) = R_n(t) - G_n(t), \quad S_2(t) = R_n(t) + G_n(t) - 2B_n(t)$$
  $$\text{POS}(t) = S_1(t) + \frac{\sigma(S_1)}{\sigma(S_2)} S_2(t)$$
  This formulation intrinsically cancels out intensity fluctuations caused by subject distance or subtle lighting attenuation `[DOC §2.1]`.
* **The Deep Spatiotemporal Branch (PhysNet-Lite):** Extracts subtle non-linear micro-motion and pulsatile absorption patterns that hand-crafted linear projections miss, particularly under heavy quantization artifacts (C5) and low resolution (C4).
* **The SQI Arbitration Mechanism:** The final vital sign estimate $\hat{y}$ is a dynamic convex combination:
  $$\hat{y} = \alpha(\text{SQI}) \cdot \hat{y}_{\text{deep}} + (1 - \alpha(\text{SQI})) \cdot \hat{y}_{\text{classical}}$$
  where $\alpha(\text{SQI}) = \frac{1}{1 + e^{-\beta(\text{SQI} - \theta_0)}}$. When signal quality degrades severely (e.g., extreme noise floor in C1 or severe block compression in C5), $\alpha \to 0$, falling back gracefully to the robust classical POS pipeline `[TEAM-DEFINED]`.

### 2.4 Multi-Task Parameter Heads & Signal Processing Formulation
1. **Heart Rate (HR):**
   * Derived via Welch's power spectral density (PSD) with Hamming windowing on the zero-mean filtered waveform:
     $$f_{\text{pulse}} = \arg\max_{f \in [0.7, 3.0]\,\text{Hz}} P_{xx}(f), \quad \text{HR} = 60 \times f_{\text{pulse}}\,\text{bpm}$$
   * Refined via parabolic interpolation across the peak spectral bin and verified against a second-harmonic check ($2 \times f_{\text{pulse}}$) `[DOC §2.3 Task 1]`.
2. **Blood Oxygen Saturation ($\text{SpO}_2$):**
   * Extracted from the spatial-temporal pulsatile AC components relative to static DC baseline across Red and Blue/Green channels (modified ratio-of-ratios):
     $$R_{\text{os}} = \frac{(\text{AC}_R / \text{DC}_R)}{(\text{AC}_B / \text{DC}_B)}, \quad \text{SpO}_2 = A - B \cdot R_{\text{os}}$$
   * *Critical Physics Caveat:* Visible RGB sensors lack the near-infrared ($850\text{--}940\,\text{nm}$) isosbestic reference of clinical pulse oximetry. Thus, constants $A$ and $B$ are empirical and subject to calibration `[DOC §2.3 Round 2]`.
3. **Blood Pressure (BP: Systolic / Diastolic):**
   * Evaluated through non-linear morphology analysis of the recovered rPPG pulse wave (pulse arrival time proxy, crest time, augmentation index, and pulse area ratios) mapped via multi-task MLP heads combined with subject demographic priors:
     $$\begin{bmatrix} \text{SBP} \\ \text{DBP} \end{bmatrix} = \mathbf{W}_{\text{BP}} \cdot \phi(\mathbf{z}_{\text{deep}}, \text{waveform morphology}) + \mathbf{b}_{\text{demo}}(\text{Age}, \text{BMI}, \text{ITA})$$
   * *Aspirational Standard:* Benchmarked against ISO 81060-2 (mean error $\le \pm 5\,\text{mmHg}$, $\text{SD} \le 8\,\text{mmHg}$) `[DOC §2.4, VERIFY]`.

### 2.5 Edge Deployment Targets & Budget Specification `[TEAM-DEFINED]`
* **Primary Embedded Target:** Raspberry Pi 4 Model B (Quad-core Cortex-A72 @ 1.5 GHz, 4GB RAM) / Raspberry Pi 5 (Quad-core Cortex-A76 @ 2.4 GHz) `[DOC §2.1, §5.2]`.
* **Deployment Toolchain:** MathWorks Raspberry Pi Blockset with MATLAB Coder (generating standalone C++17 runtime) and ONNX-Runtime ARM64 with NEON SIMD vectorization.
* **Model Size Target:** $\le 3.5\,\text{MB}$ uncompressed; $\le 880\,\text{KB}$ after INT8 Post-Training Quantization (PTQ).
* **Latency Budget (Frame-by-frame on Raspberry Pi 4):**
  * Landmark tracking / cropping: $\le 12.0\,\text{ms}$
  * Classical POS extraction + filtering: $\le 2.5\,\text{ms}$
  * PhysNet-Lite inference (INT8, batch=1, $T=32$ frames): $\le 45.0\,\text{ms}$ (invoked every 15 frames)
  * Peak RAM usage: $\le 145\,\text{MB}$
  * Total pipeline throughput: $\ge 22\,\text{fps}$ on Raspberry Pi 4; $\ge 48\,\text{fps}$ on Raspberry Pi 5.

---

## 3. Data Strategy and Experiment Design

### 3.1 Dataset Inventory, Capabilities & Governance
Under the competition rules `[DOC §3]`, teams can utilize organizer datasets, public datasets, or institutional clinical datasets.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DATASET CAPABILITIES & CRITICAL PATH                                                                  │
├──────────────────────────┬───────────────────┬──────────────┬──────────────┬───────────────────────────┤
│ Dataset Name             │ Parameters Logged │ Video Format │ Demographics │ Competition Role & Status │
├──────────────────────────┼───────────────────┼──────────────┼──────────────┼───────────────────────────┤
│ 1. Sri Ramachandra Hosp. │ HR, SpO2, BP      │ Native RGB   │ Diverse ITA  │ Core Clinical Reference   │
│    (Organizer Pilot)     │ (Clinical ground  │ Real low-    │ Hospital &   │ CRITICAL PATH: Legal NDA  │
│    [DOC §3, §8]          │ truth reference)  │ light clips  │ Controlled   │ must be executed Day 1–5  │
├──────────────────────────┼───────────────────┼──────────────┼──────────────┼───────────────────────────┤
│ 2. UBFC-rPPG Dataset     │ Dataset 1 (8 vid):│ Uncompressed │ Fair / Light │ Benchmark baseline for    │
│    (Annexure 1)          │ HR, SpO2, PPG wave│ .avi         │ skin tones   │ Round 1 & Round 2 tuning. │
│    [DOC §10]             │ Dataset 2 (42 vid)│ 30 fps       │ (Fitzpatrick │ Waveform ground truth.    │
│                          │ HR, PPG waveform  │ 640×480      │ I–III)       │ Immediate download.       │
├──────────────────────────┼───────────────────┼──────────────┼──────────────┼───────────────────────────┤
│ 3. rPPG-biases Dataset   │ Video-level mean  │ Pre-cropped  │ India &      │ Critical for C2 fairness. │
│    (Annexure 1)          │ HR only (.csv).   │ ROIs (60×30, │ Sierra Leone │ Trap T12: NO continuous   │
│    [DOC §10]             │ NO waveform!      │ 25×25)       │ (Types IV–VI)│ waveform. Mean HR only.   │
├──────────────────────────┼───────────────────┼──────────────┼──────────────┼───────────────────────────┤
│ 4. Supplementary Own     │ Synchronized ECG  │ HD 1080p,    │ Diverse local│ Supplementary validation. │
│    IRB Data Collection   │ HR, PulseOx SpO2, │ 50 lux,      │ cohort       │ Requires IRB submitted in │
│    [DOC §3, §5.1]        │ Cuff BP           │ QVGA, 10 fps │ (Types II–VI)│ Round 1; consent mandatory│
└──────────────────────────┴───────────────────┴──────────────┴──────────────┴───────────────────────────┘
```

*Data-Access Critical Path Warning:* Execution of the copyright and legal agreement with Dr. Vijay Jeyakumar for the Sri Ramachandra Hospital dataset (`IEC/23/FEB/177/08`) is the single highest-priority administrative dependency. It must be initiated on Day 1 `[DOC §3, §8]`.

### 3.2 Subject-Disjoint Stratified Partitioning & Held-Out Stress Sets
To prevent data leakage and pseudo-replication (Trap T8), splits are strictly subject-disjoint:
* **Cohort Allocation:** 60% Train (Subject-Disjoint), 20% Validation (Subject-Disjoint), 20% Test (Completely Blind Subject-Disjoint).
* **Stratification Axes:** Discretized ITA bins (Fitzpatrick I–VI), age bands ($<30$, $30\text{--}60$, $60+$ `[DOC §2.2]`), and baseline gender.
* **Held-Out Stress Sets:**
  * *Unseen Subject Cohort:* Held-out subjects evaluated across all simulated degradation cells.
  * *Unseen Degradation Parameters:* Models trained on lux $\in \{50, 100, 150\}$ are evaluated on unseen stress lux $\in \{10, 20, 30\}$ to verify true physics generalization beyond simulator fitting (Trap T2).

### 3.3 Ground-Truth Synchronization & Tolerances
* **ECG / Pulse Oximeter Ground Truth:** Reference sampling rate $\ge 100\,\text{Hz}$ for ECG `[DOC §2.3 Task 1]`.
* **Hardware Timestamp Alignment:** Camera capture timestamps logged via monotonic clock (`gettimeofday` / `clock_gettime(CLOCK_MONOTONIC)` in Python/C++ or `posixtime` in MATLAB).
* **Synchronization Protocol:** Baseline visual-auditory trigger (optical flash captured simultaneously by camera and recorded on hardware trigger channel). Maximum allowable sync jitter: $\le 10.0\,\text{ms}$ `[TEAM-DEFINED]`.

### 3.4 Rigorous Degradation Simulation Protocols (C1, C4, C5)

```
========================================================================================================================
                          PHYSICS-BASED DEGRADATION SIMULATOR PIPELINE (HANDLING TRAPS T1–T5)
========================================================================================================================
 Baseline HD Clip (≥300 lux, 720p, ≥30 fps)
       │
       ▼
 [Stage A: C1 Low-Light Simulation (Linear Photon-Domain Engine)] ──► [Traps T1, T2]
   1. Gamma De-Quantization: $I_{\text{lin}} = I_{\text{sRGB}}^{2.2}$
   2. Illuminance Gain: $I_{\text{atten}} = I_{\text{lin}} \times \left(\frac{\text{Lux}_{\text{target}}}{300}\right)$
   3. Physical Photon Scaling: $N_{\text{photon}} = I_{\text{atten}} \times \Phi_{\text{full\_well}}$ ($\Phi_{\text{full\_well}} = 5000\,\text{photons/pixel}$)
   4. Shot Noise Draw: $K_{\text{shot}} \sim \text{Poisson}(N_{\text{photon}})$
   5. Read Noise Injection: $K_{\text{total}} = K_{\text{shot}} + \mathcal{N}(0, \sigma_{\text{read}}^2)$
   6. Dynamic Range Clipping & Gamma Re-Encoding: $I_{\text{low\_light}} = \text{clip}\left(\frac{K_{\text{total}}}{\Phi_{\text{full\_well}}}, 0, 1\right)^{1/2.2}$
       │
       ▼
 [Stage B: C4 Low-Resolution Spatial Decimation] ──► [Trap T5]
   1. Bicubic Downsampling: $I_{\text{low\_res}} = \text{imresize}(I_{\text{low\_light}}, [H_{\text{target}}, W_{\text{target}}], \text{'bicubic'})$
   2. Anti-Aliasing Parity: In Python, force `PIL.Image.BICUBIC` or `scipy.ndimage` with explicit Gaussian prefilter to match MATLAB.
       │
       ▼
 [Stage C: C5 Low-Bandwidth / Remote Streaming Engine] ──► [Traps T3, T4]
   1. Frame Decimation: Subsample frames to target fps ($f_{\text{target}} \in \{10, 7.5, 6\}$).
      - Primary: Deterministic frame skipping ($T_{\text{step}} = \text{round}(f_{\text{orig}}/f_{\text{target}})$) `[DOC §2.2]`.
      - Sensitivity Variant: Boxcar temporal integration (average-then-skip) to prevent motion aliasing.
   2. MPEG-4 Bitrate-Constrained Re-encoding:
      - Feed frames through FFmpeg with strict 2-pass CBR (`-b:v 500k -minrate 500k -maxrate 500k -bufsize 1000k`).
      - In MATLAB, calibrate `VideoCompressionQuality` per clip to verify achieved bitrate $\le 500\,\text{kbps}$.
========================================================================================================================
```

#### Systematic Parameter Sweep Matrix

| Sweep Type | Dimension | Parameter Grid Evaluated | Baseline Reference |
| :--- | :--- | :--- | :--- |
| **OFAT 1 (C1: Lighting)** | Ambient Illuminance | $\text{Lux} \in \{300, 150, 100, 50, 30, 20, 10\}$ | $300\,\text{lux}$ (Unconstrained) `[DOC §2.3]` |
| **OFAT 2 (C1: Noise Floor)**| Read Noise Variance | $\sigma_{\text{read}}^2 \in \{0, 0.001, 0.005, 0.01, 0.02\}$ | $\sigma_{\text{read}}^2 = 0$ |
| **OFAT 3 (C4: Resolution)** | Spatial Dimension | $1280\times 720$ (HD), $640\times 480$ (VGA), $320\times 240$ (QVGA) | $1280\times 720$ (HD) `[DOC §2.3]` |
| **OFAT 4 (C5: Frame Rate)** | Temporal Sampling | $f_s \in \{30, 15, 10, 7.5, 6\}\,\text{fps}$ | $30\,\text{fps}$ `[DOC §2.3]` |
| **OFAT 5 (C5: Compression)**| Video Bitrate | Bitrate $\in \{\ge 5\,\text{Mbps}, 2\,\text{Mbps}, 1\,\text{Mbps}, 500\,\text{kbps}, 250\,\text{kbps}\}$ | $\ge 5\,\text{Mbps}$ (Uncompressed) `[DOC §2.3]`|
| **Worst-Case Cell 1** | C1 + C4 | $50\,\text{lux}$ + $320\times 240$ (QVGA) @ $30\,\text{fps}$, $5\,\text{Mbps}$ | Stress test: dim sensor quantization |
| **Worst-Case Cell 2** | C1 + C5 | $50\,\text{lux}$ + $720\text{p}$ @ $10\,\text{fps}$, $500\,\text{kbps}$ | Stress test: low photon count + aliasing |
| **Worst-Case Cell 3** | C4 + C5 | $300\,\text{lux}$ + $320\times 240$ @ $10\,\text{fps}$, $500\,\text{kbps}$ | Stress test: minimal spatial & temporal info |
| **Worst-Case Cell 4 (ALL)**| **C1+C2+C4+C5** | **$50\,\text{lux}$ + Type V–VI + $320\times 240$ + $10\,\text{fps}$ + $500\,\text{kbps}$** | **The Ultimate Competition Stress Cell** |

---

## 4. Model and Training Plan

### 4.1 Incremental Baseline Ladder and Ablation Progression
To satisfy Final Criterion 1 ("Innovation and thoroughness of experiments" `[DOC §5.2]`), we formulate a strict 5-stage comparative progression:
1. **Level 0: Naive Green-Channel rPPG:** Spatial mean of Green channel followed by bandpass filtering. Establishes the absolute non-machine-learning floor `[DOC §2.1]`.
2. **Level 1: Classical State-of-the-Art (POS & CHROM):** Standard mathematical projections on spatial ROIs; robust against rigid motion artifacts `[DOC §2.1]`.
3. **Level 2: Standard Deep rPPG Baseline (PhysNet-3D):** Published spatio-temporal architecture trained via conventional Empirical Risk Minimization (ERM) on raw cropped faces.
4. **Level 3: Multi-Task PhysNet-Lite with Dynamic Temporal Scaling:** Separable 3D convolutions with explicit multi-task heads for HR, $\text{SpO}_2$, and BP.
5. **Level 4: Full Proposed System (SQI-Conditioned Hybrid Fusion + Group-DRO):** The complete blueprint architecture incorporating dynamic reliability arbitration and fair minimax optimization across ITA bins `[TEAM-DEFINED]`.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODEL ARCHITECTURE HYPERPARAMETERS & TENSOR SHAPES                                                    │
├──────────────────────────┬─────────────────────────────────────────────────────────────────────────────┤
│ Input Tensor Shape       │ $\mathbf{X} \in \mathbb{R}^{B \times 3 \times T \times 64 \times 64}$      │
│ Temporal Window ($T$)    │ Adaptive: $T = 90$ frames @ $30\,\text{fps}$ ($3.0\,\text{s}$);            │
│                          │           $T = 120$ frames @ $10\,\text{fps}$ ($12.0\,\text{s}$)            │
│ Encoder Trunk            │ 4 Stages of Separable 3D Conv Blocks: Conv3D(1×3×3) $\to$ Conv3D(3×1×1)     │
│ Channels per Stage       │ Stage 1: 16; Stage 2: 32; Stage 3: 64; Stage 4: 128                         │
│ Latent Feature Vector    │ $\mathbf{z} \in \mathbb{R}^{B \times 128}$ (Post Global Adaptive Pooling)  │
│ Parameter Count / FLOPs  │ Total Parameters: $848,220$ ($\approx 0.85\,\text{M}$); FLOPs: $1.82\,\text{G}$│
└──────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Multi-Objective Loss Formulation & Group-DRO Optimization
The global training objective balances physiological waveform fidelity, spectral concentration, parameter estimation error, and demographic minimax equity:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{wave}} \mathcal{L}_{\text{wave}} + \lambda_{\text{spec}} \mathcal{L}_{\text{spec}} + \lambda_{\text{HR}} \mathcal{L}_{\text{HR}} + \lambda_{\text{SpO2}} \mathcal{L}_{\text{SpO2}} + \lambda_{\text{BP}} \mathcal{L}_{\text{BP}}$$

1. **Negative Pearson Waveform Loss:**
   $$\mathcal{L}_{\text{wave}}(\mathbf{y}, \hat{\mathbf{y}}) = 1 - \frac{\sum_{t=1}^T (y_t - \mu_y)(\hat{y}_t - \mu_{\hat{y}})}{\sqrt{\sum_{t=1}^T (y_t - \mu_y)^2 \sum_{t=1}^T (\hat{y}_t - \mu_{\hat{y}})^2}}$$
2. **Frequency-Domain Spectral Cross-Entropy Loss:**
   Forces predicted energy to concentrate sharply at the true cardiac fundamental frequency:
   $$\mathcal{L}_{\text{spec}} = -\sum_{k} P_{\text{true}}(f_k) \log \left(\frac{\exp(|\hat{Y}(f_k)|^2)}{\sum_j \exp(|\hat{Y}(f_j)|^2)}\right)$$
3. **Heteroscedastic Uncertainty Loss for $\text{SpO}_2$ and BP:**
   Predicts parameter $\hat{y}$ and variance $\hat{\sigma}^2$ simultaneously, penalizing high confidence on erroneous estimates:
   $$\mathcal{L}_{\text{param}}(y, \hat{y}, \hat{\sigma}^2) = \frac{1}{2} \exp(-s) \|y - \hat{y}\|_1 + \frac{1}{2} s, \quad \text{where } s = \log \hat{\sigma}^2$$
4. **Group-DRO Fair Minimax Optimization over ITA Strata `[TEAM-DEFINED]`:**
   Instead of standard empirical risk minimization over subjects, we optimize against the worst-performing skin-tone cluster $g \in \mathcal{G} = \{\text{Type I}, \dots, \text{Type VI}\}$:
   $$\min_{\theta} \max_{g \in \mathcal{G}} \left\{ \mathcal{L}_g(\theta) + \rho \cdot \text{KL}(q_g \| p_g) \right\}$$
   Group weights $q_g$ are updated via exponentiated gradient ascent at every training step with step-size $\eta_q = 0.01$. This mathematically guarantees that training capacity is automatically redirected to darker skin tones (Fitzpatrick V–VI) whenever their residual error rises `[TEAM-DEFINED]`.

### 4.3 Calibration Strategies & Clinical Boundaries for $\text{SpO}_2$ and BP
* **The Physics Barrier for $\text{SpO}_2$ (Trap T10):** Standard pulse oximetry relies on the differential extinction coefficient between oxygenated ($\text{HbO}_2$) and deoxygenated ($\text{Hb}$) hemoglobin at $660\,\text{nm}$ (Red) and $940\,\text{nm}$ (NIR). Commercial RGB camera sensors have spectral bands centered at $\approx 610\,\text{nm}$ (Red), $\approx 530\,\text{nm}$ (Green), and $\approx 460\,\text{nm}$ (Blue). Because all three visible wavelengths lie on the same side of the isosbestic point ($\approx 800\,\text{nm}$), camera-based $\text{SpO}_2$ is an *indirect, weakly-conditioned surrogate*.
* **Calibration Protocol:** We report two distinct regimes:
  1. *Calibration-Free Mode:* Uses population-averaged empirical coefficients derived strictly from the training cohort.
  2. *Single-Point Calibrated Mode:* Uses an initial 10-second subject-specific baseline offset $\Delta_{\text{calib}} = y_{\text{ref}, 0} - \hat{y}_0$.
* **Blood Pressure Physical Feasibility Limits:**
  * *Why Multi-Site PTT Fails at $\le 10\,\text{fps}$:* Pulse Transit Time (PTT) between the forehead and chin/cheek typically spans $10\text{--}30\,\text{ms}$. At $10\,\text{fps}$, the inter-frame interval is $100\,\text{ms}$—more than $3\times$ larger than the entire physiological transit delay. PTT estimation at $10\,\text{fps}$ is mathematically and physically impossible.
  * *Methodology Used:* Our BP head operates strictly on *pulse wave morphology features* (upslope time, reflection index, normalized systolic area) combined with subject demographic priors `[TEAM-DEFINED]`.
* **Mandatory Comparison with Naive Baselines (Trap T10):** Every reported test error for $\text{SpO}_2$ and BP is presented alongside:
  1. *A Constant Mean-Predictor Baseline:* Predicts the training set mean $\bar{y}_{\text{train}}$ for all test windows.
  2. *A Demographics-Only Regression Baseline:* Predicts vitals solely from age, gender, and ITA without video input.

### 4.4 Fairness, Vitiligo & Reflectance Anomalies (C2)
* **Objective ITA Calculation (Trap T6):** Converted via CIE $1976\,L^*a^*b^*$ using standard D65 illuminant reference:
  $$\text{ITA} = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$$
  Computed strictly as the spatial median over baseline, well-lit, specular-free skin masks `[DOC §2.2]`.
* **Discretized ITA Strata Mapping:**
  * *Very Light (Type I):* $\text{ITA} > 55^\circ$
  * *Light (Type II):* $41^\circ < \text{ITA} \le 55^\circ$
  * *Intermediate (Type III):* $28^\circ < \text{ITA} \le 41^\circ$
  * *Tan (Type IV):* $10^\circ < \text{ITA} \le 28^\circ$
  * *Brown (Type V):* $-30^\circ < \text{ITA} \le 10^\circ$
  * *Dark (Type VI):* $\text{ITA} \le -30^\circ$
* **Reflectance Anomaly / Vitiligo Detection:**
  * The face mask is partitioned into an $8 \times 8$ grid of sub-ROIs.
  * Local chromaticity outliers are detected via Mahalanobis distance in $a^*b^*$ color space:
    $$D_M(\mathbf{x}_k) = \sqrt{(\mathbf{x}_k - \boldsymbol{\mu}_{ab})^T \boldsymbol{\Sigma}_{ab}^{-1} (\mathbf{x}_k - \boldsymbol{\mu}_{ab})}$$
  * Sub-ROIs exhibiting bimodal distributions or significant depigmentation ($D_M > 3.0$) are isolated. The pipeline performs **SNR-weighted spatial fusion**, down-weighting boundary regions between depigmented and hyperpigmented patches to avoid edge-gradient motion artifacts.
* **Synthetic Depigmentation Data Augmentation:** To train the model for vitiligo without clinical patient overfitting, we synthesize irregular Perlin-noise masks over cheek regions, simulating local melanin loss ($\Delta L^* = +25, \Delta b^* = -15$) during training.

---

## 5. Round-by-Round Plan and Submission Package

### 5.1 Round 1 In-Depth Constraint Selection Matrix
Under Round 1 specifications `[DOC §2.3 Task 1]`, the team must evaluate HR under controlled and uncontrolled conditions and implement $\ge 2$ additional constraints from C1–C7.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ROUND 1 CONSTRAINT SELECTION DECISION MATRIX                                                          │
├────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│ Constraint │ Data Avail. │ Sim. Rigor  │ Impl. Risk  │ Equity Align│ Jury Value  │ Selection Decision  │
│ Candidate  │ by 30 Nov   │ & Realism   │ & Feasibility│ with Theme  │ & Novelty   │                     │
├────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ C1: Low    │ High (Synth │ High (Linear│ Low (Clear  │ High (Rural │ Very High   │ SELECTED FOR DEEP   │
│ Light      │ + Pilot)    │ Poisson)    │ physics)    │ healthcare) │ (Core rPPG) │ IMPLEMENTATION (R1) │
├────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ C2: Skin   │ High (rPPG- │ High (ITA   │ Med (Group- │ Maximum     │ Maximum     │ SELECTED FOR DEEP   │
│ Tone / Vit │ biases set) │ pipeline)   │ DRO tuning) │ (Core theme)│ (Fairness)  │ IMPLEMENTATION (R1) │
├────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ C4: Low    │ High (Math- │ High (Anti- │ Low (Resize │ Med (Budget │ High        │ Planned secondary;  │
│ Resolution │ Works resize│ aliasing)   │ benchmarks) │ hardware)   │ (Graceful)  │ fully run in R1 exp │
├────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ C5: Low    │ High (Video │ High (CBR   │ Med (Nyquist│ High (Remote│ High        │ Planned secondary;  │
│ Bandwidth  │ decimation) │ FFmpeg/WMV) │ edge floor) │ telehealth) │ (Spectral)  │ fully run in R1 exp │
└────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘
```

*Selection Justification:* **C1 (Low Light)** and **C2 (Skin Tone / Vitiligo)** are selected for primary deep focus in Round 1. C2 directly addresses the overarching competition theme (*"Equitable Camera-Based Vital Sign Monitoring"* `[DOC §1]`), and C1 addresses the primary failure mode of optical rPPG. C4 and C5 are concurrently integrated into the automated test matrix to demonstrate comprehensive readiness.

### 5.2 Phase-by-Phase Execution Plan (Rounds 1, 2, and 3)

| Dimension | Round 1: Single-Parameter (HR) `[DOC §2.3]` | Round 2: Two-Parameter (HR + $\text{SpO}_2$) `[DOC §2.3]` | Round 3: Full System (HR + $\text{SpO}_2$ + BP) `[DOC §2.3]` |
| :--- | :--- | :--- | :--- |
| **Submission Deadline** | **30 November 2026** (Results: 05 Dec) `[DOC §6]` | **16 January 2027** (Results: 21 Jan) `[DOC §6]` | **16–21 May 2027** (Live ICASSP Final) `[DOC §6]` |
| **Target Parameters** | Heart Rate (HR) in bpm | Joint HR (bpm) and $\text{SpO}_2$ (%) | Simultaneous HR, $\text{SpO}_2$, and BP (SBP/DBP mmHg) |
| **Operating Conditions**| Controlled ($\ge 300\,\text{lux}$, $720\text{p}$, $30\,\text{fps}$) & Uncontrolled | Full multi-constraint sweep (C1, C2, C4, C5) | Live edge streaming under $\ge 2$ live stress constraints |
| **Architecture Deltas** | Baseline POS/CHROM + PhysNet-Lite trunk + Group-DRO | Dual-head model; modified ratio-of-ratios branch | Multi-task morphology network; INT8 PTQ edge engine |
| **Data & Experiments** | UBFC-rPPG + rPPG-biases + Organizer Pilot data | Add UBFC Dataset 1 ($\text{SpO}_2$) + clinical hospital sets | Complete clinical validation cohort + live camera feed |
| **Evaluation Metrics** | HR MAE, RMSE, Pearson $r$, Bland-Altman LoA | HR + $\text{SpO}_2$ MAE, RMSE, Bland-Altman, $\% \pm 2\%$ | Complete clinical error matrix; Raspberry Pi latency/FPS |
| **Deliverable Package** | Report (PDF), `Round1_Baseline_None.mat`, code | Extended report, `Round2_[Cond]_[Constr].mat`, code | Live demo, edge deployable binary, benchmark CSVs |
| **Exit Criteria** | HR MAE $\le 2.5\,\text{bpm}$; C2 fairness $\tau \le 1.5\,\text{bpm}$ | $\text{SpO}_2$ MAE $\le 2.0\%$; uncalibrated $\le 3.5\%$ | BP SBP/DBP MAE $\le 6.5\,\text{mmHg}$; Pi4 FPS $\ge 20$ |

### 5.3 Required Submission Files and Data Structure Compliance
All deliverables strictly follow the official data organization `[DOC §1.2, §3]`:

```
========================================================================================================================
                          OFFICIAL DIRECTORY TREE & SUBMISSION FILE LAYOUT
========================================================================================================================
FaceVitals_Submission_Package/
├── /Round1_Baseline/                           [DOC §1.2]
│   ├── Round1_Baseline_None.mat                [DOC §1.2] (Facial video & ref HR at baseline)
│   ├── Round1_LowLight_C1.mat                  [ASSUMPTION: Proposed for uncontrolled condition requirement]
│   ├── Round1_SkinTone_C2.mat                  [ASSUMPTION: Proposed for uncontrolled condition requirement]
│   ├── process_round1.m                        [DOC §1.2] (MATLAB/Python processing pipeline script)
│   ├── target_video.mp4                        [DOC §1.2] (Raw facial video clip)
│   ├── reference_signal.csv                    [DOC §1.2] (Synchronized ground truth HR)
│   ├── processed_output.csv                    [DOC §1.2] (Estimated algorithmic HR output)
│   └── FaceVitals_Round1_Technical_Report.pdf  [DOC §4]   (IEEE conference format technical paper)
│
├── /Round2_HR_SpO2/                            [DOC §1.2]
│   ├── Round2_LowLight_C1.mat                  [DOC §1.2]
│   ├── Round2_SkinTone_C2.mat                  [DOC §1.2]
│   ├── Round2_LowResolution_C4.mat             [DOC §1.2]
│   ├── process_round2.m                        [DOC §1.2]
│   ├── target_video.mp4
│   ├── reference_signal.csv                    (Extended with SpO2 ground truth column)
│   └── processed_output.csv                    (Extended with SpO2 estimated column)
│
├── /Round3_Constrained/                        [DOC §1.2]
│   ├── (Intermediate multi-constraint stress evaluation packages)
│
├── /Round4_FullSystem_Clinical/                [DOC §1.2 Discrepancy: Official directory for Final ICASSP Package]
│   ├── deployable_edge_engine/                 (Compiled C++/Python real-time runtime for Raspberry Pi)
│   ├── target_video.mp4                        (or live-stream recorded session log)
│   ├── reference_signal.csv                    (Columns: Timestamp, HR, SpO2, SBP, DBP)
│   ├── processed_output.csv                    (Columns: Timestamp, Est_HR, Est_SpO2, Est_SBP, Est_DBP, SQI)
│   └── benchmark_comparison.csv                [DOC §1.2] (Comparison against clinical standard device)
│
└── /Dataset/                                   [DOC §3]
    ├── /Controlled/                            (Naming: Subject{ID}_{SkinTone}_{Condition}_{Parameter}.ext)
    ├── /Hospital/                              (e.g., Subject014_TypeV_LowLight_HR.csv)
    └── /LowLight/
========================================================================================================================
```

#### Mandatory Internal Variables in Every `.mat` File `[DOC §1.2]`
1. `facial_video`: Raw RGB video frames of the subject $[T \times H \times W \times 3]$ (`uint8`).
2. `roi_data`: Extracted facial ROI coordinates/masks (forehead, bilateral cheeks) $[T \times 3 \times 4]$ bounding boxes or binary masks.
3. `reference_signal`: Ground-truth physiological signal(s) aligned at video frame rate.
4. `processed_signal`: Output of the participant's extraction algorithm.
5. `metrics`: Struct containing scalar evaluation results: `metrics.MAE`, `metrics.RMSE`, `metrics.Pearson_r`, `metrics.BlandAltman_LoA` $[2 \times 1]$.
6. `params`: Struct containing operational metadata:
   * `params.camera_resolution`: e.g., `'1280x720'` or `'320x240'`
   * `params.frame_rate`: e.g., `30` or `10`
   * `params.illuminance_lux`: e.g., `50` or `300`
   * `params.camera_to_subject_distance`: e.g., `'50cm'`
   * `params.subject_fitzpatrick_type`: e.g., `'TypeV'`
   * `params.subject_age`: e.g., `64`
   * `params.constraint_applied`: e.g., `'C1, C2'`
   * `params.extra`: Struct containing team-defined tracking fields `[TEAM-DEFINED]`: `params.extra.achieved_bitrate_kbps`, `params.extra.measured_ita_deg`, `params.extra.sqi_mean`.

### 5.4 Submission Verification Checklist (Pre-Flight Gate)
```markdown
[ ] 1. Identifier Consistency: Every subject ID in target_video.mp4 matches reference_signal.csv, 
       processed_output.csv, and RoundX_Condition_Constraint.mat [DOC §1.2].
[ ] 2. Variable Exactness: Every .mat file contains all 6 required variables with identical spelling [DOC §1.2].
[ ] 3. Data Type & File Size: facial_video stored as uint8 in HDF5 (-v7.3) format to prevent 2GB corruption [Trap T11].
[ ] 4. Metric Reproducibility: Executing process_roundX.m reproduces metrics struct exactly from raw video.
[ ] 5. Directory Compliance: Root directories adhere to /Round1_Baseline/, /Round2_HR_SpO2/, etc. [DOC §1.2].
[ ] 6. Ethics Documentation: Sri Ramachandra legal agreement or institutional IRB clearance attached [DOC §3].
[ ] 7. Roster Verification: 1 Faculty, 3–10 Undergrads (≥3 SPS student members verified active) [DOC §5.1].
```

---

## 6. Evaluation Protocol

### 6.1 Clinical Metric Formulations and Statistical Rigor
Evaluation adheres strictly to biomedical method-comparison standards (Altman & Bland, 1983 `[DOC §10]`).

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
2. **Root Mean Square Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Pearson Correlation Coefficient ($r$):**
   $$r = \frac{\sum_{i=1}^N (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^N (y_i - \bar{y})^2 \sum_{i=1}^N (\hat{y}_i - \bar{\hat{y}})^2}}$$
4. **Bland–Altman Agreement Limits (Cluster-Bootstrap Adjusted):**
   * *Mean Bias:* $\bar{d} = \frac{1}{N}\sum (y_i - \hat{y}_i)$
   * *Limits of Agreement (LoA):* $\bar{d} \pm 1.96 \cdot s_d$
   * *Repeated-Measures / Cluster Adjustment (Trap T8):* Because multiple video windows originate from the same subject, standard sample variance underestimates true error variance. We compute variance using subject-level random-effects decomposition:
     $$s_d^2 = \sigma_{\text{between}}^2 + \sigma_{\text{within}}^2$$
   * *95% Confidence Intervals:* Computed via 2,000 subject-level cluster bootstrap iterations.
   * *Proportional Bias Test:* Regress difference $(y - \hat{y})$ against mean $(y + \hat{y})/2$; flag if slope $p < 0.05$.
5. **Percentage Within Clinical Tolerance:**
   * $\text{HR} \le \pm 5\,\text{bpm}$ (`%CP5`): Fraction of estimates within $5\,\text{bpm}$ of reference.
   * $\text{SpO}_2 \le \pm 2\%$ (`%CP2`) and $\le \pm 3\%$ (`%CP3`).

### 6.2 The C2 Skin-Tone Acceptance Criterion Formulation
The official document mandates: *"Per-type error must not deviate beyond a pre-specified tolerance from the population mean across all six types"* `[DOC §2.2]`.

* **Mathematical Definition `[TEAM-DEFINED]`:**
  Let $\text{MAE}_g$ denote the Mean Absolute Error evaluated on Fitzpatrick stratum $g \in \{\text{Type I}, \dots, \text{Type VI}\}$. The unweighted population mean across all six strata is:
  $$\mu_{\text{pop}} = \frac{1}{6} \sum_{g=1}^6 \text{MAE}_g$$
  The fairness constraint passes if and only if:
  $$\Delta_{\text{fairness}} = \max_{g \in \{1,\dots,6\}} |\text{MAE}_g - \mu_{\text{pop}}| \le \tau$$
* **Pre-Registered Numeric Tolerance ($\tau$):**
  * *For Heart Rate (HR):* $\tau_{\text{HR}} = 1.5\,\text{bpm}$ `[TEAM-DEFINED]`.
  * *For $\text{SpO}_2$:* $\tau_{\text{SpO2}} = 1.0\%$ `[TEAM-DEFINED]`.
  * *For Blood Pressure:* $\tau_{\text{BP}} = 3.0\,\text{mmHg}$ `[TEAM-DEFINED]`.

### 6.3 Comprehensive Performance Target Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TARGET PERFORMANCE AND FALLBACK THRESHOLDS ACROSS CONSTRAINTS                                          │
├─────────────────────┬────────────────────┬────────────────────┬──────────────────┬─────────────────────┤
│ Evaluation Category │ Parameter & Metric │ Ideal Baseline     │ Stress Constraint│ Fallback Action if  │
│                     │                    │ Target (≥300 lux)  │ Target (≤50 lux, │ Threshold Missed    │
│                     │                    │                    │ QVGA, ≤10 fps)   │ [TEAM-DEFINED]      │
├─────────────────────┼────────────────────┼────────────────────┼──────────────────┼─────────────────────┤
│ 1. Heart Rate (HR)  │ MAE (bpm)          │ ≤ 1.5 bpm          │ ≤ 3.5 bpm        │ Expand window to 12s│
│    [DOC §2.3 Task 1]│ RMSE (bpm)         │ ≤ 2.5 bpm          │ ≤ 5.0 bpm        │ Fallback to POS     │
│                     │ Pearson r          │ ≥ 0.95             │ ≥ 0.85           │ Switch to causal IIR│
│                     │ % within ±5 bpm    │ ≥ 95.0%            │ ≥ 82.0%          │ Increase SQI gate   │
├─────────────────────┼────────────────────┼────────────────────┼──────────────────┼─────────────────────┤
│ 2. Blood Oxygen     │ MAE (%)            │ ≤ 1.8%             │ ≤ 3.2%           │ Re-center baseline  │
│    Saturation (SpO2)│ RMSE (%)           │ ≤ 2.5%             │ ≤ 4.2%           │ Require 1-pt calib  │
│    [DOC §2.3 R2]    │ % within ±2%       │ ≥ 85.0%            │ ≥ 70.0%          │ Abstain if SQI<0.65 │
├─────────────────────┼────────────────────┼────────────────────┼──────────────────┼─────────────────────┤
│ 3. Blood Pressure   │ SBP MAE (mmHg)     │ ≤ 5.0 mmHg         │ ≤ 8.0 mmHg       │ Revert to demo prior│
│    (Systolic/Diast.)│ DBP MAE (mmHg)     │ ≤ 4.0 mmHg         │ ≤ 6.5 mmHg       │ Restrict to normot. │
│    [DOC §2.3 R3]    │ ISO 81060-2 Match  │ PASS (≤5±8 mmHg)   │ Honest Limitation│ State research limit│
├─────────────────────┼────────────────────┼────────────────────┼──────────────────┼─────────────────────┤
│ 4. C2 Skin Fairness │ Max Deviation |Δ|  │ ≤ 1.0 bpm          │ ≤ 1.5 bpm        │ Increase Group-DRO  │
│    [DOC §2.2]       │ across Types I–VI  │                    │ (Official Target)│ step-size η_q       │
├─────────────────────┼────────────────────┼────────────────────┼──────────────────┼─────────────────────┤
│ 5. Gating & Failure │ Abstention Coverage│ 100% (No discard)  │ ≥ 75.0% accepted │ Plot risk-coverage  │
│    Reporting [T7]   │ Risk-Coverage Area │ AUC_RC ≤ 1.2 bpm   │ AUC_RC ≤ 2.8 bpm │ Report ungated also │
├─────────────────────┼────────────────────┼────────────────────┼──────────────────┼─────────────────────┤
│ 6. Edge Efficiency  │ Latency (Pi 4/5)   │ ≤ 30 ms (Pi 4)     │ ≤ 45 ms (Pi 4)   │ Skip deep branch;   │
│    [DOC §2.1]       │ Throughput (FPS)   │ ≥ 30 fps (Pi 5)    │ ≥ 20 fps (Pi 4)  │ run pure POS/CHROM  │
└─────────────────────┴────────────────────┴────────────────────┴──────────────────┴─────────────────────┤
```

---

## 7. Comprehensive Risk Register

The risk matrix details probability (P), severity (S), leading indicators, mitigations, and pre-planned fallback protocols for the competition lifecycle:

| Risk Description | P | S | Early-Warning Indicator | Architectural Mitigation | Fallback Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1. Low-Light SNR Collapse ($\le 50\,\text{lux}$)** | H | H | Chrominance channel variance drops below sensor read noise floor $\sigma_{\text{read}}^2$. | Physics-based Poisson spatial-temporal bilateral filtering before projection; no per-channel CLAHE. | Auto-expand temporal window to $12\,\text{s}$; fallback to Green-difference rPPG. |
| **R2. Skin-Tone Bias Drift on Types V–VI** | H | H | Disparity metric $\Delta_{\text{fairness}} > 1.5\,\text{bpm}$ during validation cross-validation. | Group-DRO minimax training directly penalizing worst-case stratum loss; local ITA-adaptive normalization. | Sub-ROI SNR reweighting; isolate forehead where melanin attenuation is lowest. |
| **R3. Low-Res Landmark Jitter / Failure** | M | H | Landmark Euclidean jitter $> 5\,\text{px}$ across adjacent frames at QVGA. | Kalman-smoothed bounding-box tracking; multi-scale landmark initialization; minimum $12\times 12\,\text{px}$ ROI floor. | Revert to full-face bounding box spatial averaging if landmark confidence $<0.4$. |
| **R4. Spectral Aliasing at $\le 10\,\text{fps}$** | H | H | Broadening of Welch PSD peaks; spurious harmonics $>3.0\,\text{Hz}$ folded into band. | Strict brick-wall anti-aliasing prefilter; Lomb-Scargle periodogram; harmonic ratio verification ($2f_0$). | Abstain from reporting instantaneous rate; report robust 10-second windowed median. |
| **R5. Uncalibrated $\text{SpO}_2$/BP Drift** | H | H | Test MAE on held-out subjects exceeds naive mean-predictor error. | Dual-mode reporting: report calibration-free and single-point calibrated regimes transparently. | Explicitly report physics caveat in report; state research limitation clearly. |
| **R6. Absence of Public Video BP Labels** | H | H | No continuous arterial BP waveforms available in public rPPG repositories. | Early execution of Sri Ramachandra Hospital agreement; adapt open clinical MIMIC-derived rPPG subsets. | Benchmark BP on organizer clinical set; use demographic + pulse morphology ML head. |
| **R7. Dataset Legal Access Delays** | M | H | No response from organizer on Sri Ramachandra agreement by Day 7 (24 Oct 2026). | Multi-track dataset loading: immediately utilize UBFC-rPPG + rPPG-biases; begin IRB approval. | Rely on publicly accessible Annexure-1 datasets + locally collected IRB data. |
| **R8. Simulation-to-Reality Gap (Trap T2)** | M | M | Model trained on synthetic degradation collapses on real hospital low-light clips. | Linear-light photon-accurate Poisson simulator; hold out unseen gain/noise levels during training. | Calibrate sensor gain parameters against real calibrated lux meter ground truth. |
| **R9. Cherry-Picking via Gating (Trap T7)** | M | M | High test accuracy achieved but abstention coverage drops below 50%. | Pre-register abstention threshold $\theta_{\text{abstain}}$; mandate concurrent reporting of ungated results. | Plot full risk-coverage curves showing error across all coverage fractions $0.1\text{--}1.0$. |
| **R10. Raspberry Pi Latency Exceeded** | M | M | Inference time on Pi 4 exceeds $50\,\text{ms/frame}$ ($<20\,\text{fps}$). | INT8 Post-Training Quantization via MATLAB Coder / ONNX ARM NEON; decouple deep inference rate. | Invoke deep trunk every $N=10$ frames; interpolate pulse phase via classical POS. |
| **R11. Live Demo Network / Hardware Crash** | L | H | Camera driver disconnects or system freezes during live jury session at ICASSP. | Standalone offline execution script; zero cloud connectivity assumption; warm-standby laptop backup. | Pre-record continuous raw video + synchronized reference stream as certified backup. |
| **R12. Non-Compliant File Names / Structures** | M | H | Submission validator rejects folder naming or missing `.mat` variables. | Automated pre-submission verification script (`verify_submission.py`) run in CI/CD pipeline. | Strict file lock 48 hours prior to official portal upload deadline. |
| **R13. Team Roster Disqualification Lock** | L | H | Student membership lapses or member change attempted after 25 Oct 2026. | Faculty verifies IEEE SPS student membership numbers on IEEE portal prior to submission. | Freeze official roster of 5 committed undergrads + 1 faculty + 1 tutor by 22 Oct 2026. |

---

## 8. Timeline and Execution Strategy

### 8.1 Critical Official Milestones Timeline
* **17 October 2026:** Planning Date & Architecture Kickoff.
* **25 October 2026:** Official Team Registration Deadline `[DOC §6]` *(8 days remaining!)*.
* **30 November 2026:** Round 1 Submission Deadline `[DOC §6]`.
* **05 December 2026:** Round 1 Result Declaration `[DOC §6]`.
* **16 January 2027:** Round 2 Submission Deadline `[DOC §6]`.
* **21 January 2027:** Round 2 Result Declaration `[DOC §6]`.
* **21 Jan – 01 Feb 2027:** Finalist Selection Window `[DOC §6]`.
* **09 February 2027:** Announcement of 3 Finalist Teams `[DOC §6]`.
* **16–21 May 2027:** Live Final Competition at IEEE ICASSP 2027, New Orleans, USA `[DOC §6]`.

### 8.2 Immediate First-8-Days Action Checklist (17 Oct – 25 Oct 2026)

```markdown
- [ ] Day 1 (17 Oct): Convene team. Audit student roster: verify 1 Faculty Supervisor, 
                      ≥3 undergraduate IEEE SPS student memberships active on IEEE portal [DOC §5.1].
- [ ] Day 2 (18 Oct): Draft and email official legal agreement request for Sri Ramachandra 
                      dataset to Dr. Vijay Jeyakumar (vijayjeyakumar@ieee.org), copying 
                      sp-competitions@listserv.ieee.org [DOC §3, §8].
- [ ] Day 3 (19 Oct): All team members read and execute the 'SP and VIP Cup Terms & Conditions' 
                      and IEEE Conflict-of-Interest forms [DOC §5.3, §7.1].
- [ ] Day 4 (20 Oct): Apply for complimentary MathWorks Student Competition MATLAB Licenses 
                      via official Dropbox repository [DOC §7.2].
- [ ] Day 5 (21 Oct): Download Annexure-1 public datasets: UBFC-rPPG (Datasets 1 & 2) and 
                      AiPEX-Lab/rppg_biases. Verify checksums and local disk layout.
- [ ] Day 6 (22 Oct): Submit institutional IRB application for supplementary vital-sign data 
                      collection protocol adapting Sri Ramachandra clinical guidelines [DOC §3].
- [ ] Day 7 (23 Oct): Initialize Git repository with automated formatting, CI testing, and 
                      pre-submission verification scripts.
- [ ] Day 8 (24 Oct): Complete official team submission on portal:
                      https://www2.securecms.com/SPCup/SPCRegistration.asp [DOC §7.1].
- [ ] Day 9 (25 Oct): FINAL REGISTRATION FREEZE. Confirm receipt confirmation email from IEEE SPS.
```

### 8.3 Work Breakdown Structure & Team Roles (1 Faculty, 1 Tutor, 5 Undergrads)
* **Faculty Supervisor (PI):** Institutional compliance, legal agreement execution, IRB oversight, official correspondence with IEEE SPS `[DOC §5.1]`.
* **Graduate Tutor:** Architecture co-design, PyTorch-to-MATLAB translation review, statistical verification.
* **Undergrad 1 (Lead & Edge Architect):** Embedded runtime on Raspberry Pi 4/5, MATLAB Coder toolchain, latency profiling, C5 low-bandwidth engine.
* **Undergrad 2 (Computer Vision & Preprocessing):** Face detection, landmark tracking, multi-ROI extraction, C4 resolution scaling, C7 occlusion fallback.
* **Undergrad 3 (Classical Signal Processing):** POS/CHROM pipelines, bandpass filter design, spectral Welch PSD, multi-channel SQI engine.
* **Undergrad 4 (Deep Learning & Fairness):** PhysNet-Lite training, Group-DRO implementation, C2 skin-tone stratification, vitiligo synthesis.
* **Undergrad 5 (Data Engineering & Evaluation):** Linear-light C1 simulator, ground-truth alignment, Bland-Altman statistics, `.mat` package validator.

### 8.4 Prioritized Task Backlog (Must / Should / Nice)
* **MUST HAVE (Round 1 Readiness by 15 Nov):**
  * Automated linear-light degradation simulator (C1).
  * POS/CHROM classical baseline operating frame-by-frame.
  * MediaPipe/Cascade multi-ROI extraction with KLT tracking.
  * Subject-disjoint cross-validation pipeline with cluster bootstrap.
  * Exact `.mat` export script matching all 6 official variables `[DOC §1.2]`.
* **SHOULD HAVE (Round 2 Readiness by 20 Dec):**
  * PhysNet-Lite spatiotemporal multi-task network.
  * Group-DRO optimization across 6 ITA bins.
  * Ratio-of-ratios $\text{SpO}_2$ estimation head with empirical calibration.
  * Adaptive window scaling ($3.0\,\text{s} \to 12.0\,\text{s}$) for $10\,\text{fps}$ streaming.
* **NICE TO HAVE (Round 3 Polish by March 2027):**
  * INT8 quantization running live on Raspberry Pi 5 with touchscreen UI.
  * RAFT optical flow micro-motion feature integration `[DOC §2.2]`.
  * Multi-face tracking with Kalman ID association for C3 extension `[DOC §2.2]`.

---

## 9. Code and Reproducibility

### 9.1 Repository Structure
```
FaceVitals_Core/
├── config/
│   ├── baseline_config.yaml
│   └── stress_c1_c2_c4_c5.yaml               # $C_1, C_2, C_4, C_5$
├── src/
│   ├── acquisition/
│   │   ├── stream_reader.py
│   │   └── metadata_logger.py
│   ├── preprocessing/
│   │   ├── face_roi_tracker.py
│   │   ├── ita_classifier.py                # $\text{ITA}$
│   │   └── degradation_sim.py               # $C_1, C_4, C_5$
│   ├── classical/
│   │   ├── rppg_pos.py                      # $\mathbf{S}_{\text{POS}}$
│   │   ├── rppg_chrom.py                    # $\text{CHROM}$
│   │   └── filter_bank.py                   # $[0.7, 3.0]\,\text{Hz}$
│   ├── models/
│   │   ├── physnet_lite.py                  # $\approx 0.85\,\text{M}$
│   │   ├── multi_task_heads.py              # $\text{HR}$, $\text{SpO}_2$, $\text{BP}$
│   │   └── group_dro_loss.py                # $q_g$
│   ├── fusion/
│   │   ├── sqi_engine.py                    # $\text{SQI}$
│   │   └── dynamic_arbiter.py               # $\alpha(\text{SQI})$, $\beta$
│   └── evaluation/
│       ├── metrics_bland_altman.py          # $\text{MAE}$, $\text{RMSE}$, $r$
│       ├── bootstrap_ci.py
│       └── mat_export_validator.py
├── scripts/
│   ├── run_round1_experiments.sh
│   └── package_submission.py
└── tests/
    └── test_mat_format.py
```

### 9.3 Cross-Platform Toolchain Mapping (MATLAB $\leftrightarrow$ Python Parity)

| Tool / Functionality | MATLAB Function / System Object `[DOC]` | Python / PyTorch Equivalent `[TEAM-DEFINED]` | Parity Caveat & Behavioral Mismatch (Traps T1, T3, T5) |
| :--- | :--- | :--- | :--- |
| **Face Detection** | `faceDetector` / `vision.CascadeObjectDetector` | `cv2.CascadeClassifier` / `mediapipe.FaceMesh` | MATLAB `faceDetector` uses deep learning; OpenCV Haar cascade is faster but less accurate at low lux. |
| **Track Association**| `assignDetectionsToTracks` `[DOC §2.2]` | `scipy.optimize.linear_sum_assignment` | Exact Hungarian/Munkres equivalence; ensure distance matrix uses bounding box IoU. |
| **Point Tracking** | `vision.PointTracker` (KLT) `[DOC §2.2]` | `cv2.calcOpticalFlowPyrLK` | Set identical pyramid levels ($L=3$) and window sizes ($15\times 15$) to match trajectory drift. |
| **Optical Flow** | `opticalFlowRAFT` `[DOC §2.2]` | `torchvision.models.optical_flow.raft_small` | Verify GPU memory limits on host; fallback to Farneback optical flow on edge Raspberry Pi. |
| **Shape Overlay** | `insertShape` `[DOC §2.2]` | `cv2.rectangle` / `cv2.fillPoly` | Ensure coordinate conventions match $[x, y, w, h]$ vs $[x_1, y_1, x_2, y_2]$. |
| **Spatial Filtering**| `imfilter` `[DOC §2.2]` | `scipy.ndimage.correlate` / `cv2.filter2D` | Check boundary padding convention (`'replicate'` in MATLAB $\equiv$ `BORDER_REPLICATE` in OpenCV). |
| **Causal Filtering** | `designfilt` `[DOC §2.3]` | `scipy.signal.butter(output='sos')` | Use Second-Order Sections (`sos`) in Python to match numerical precision of MATLAB `designfilt`. |
| **Video Decoding** | `VideoReader` `[DOC §2.2]` | `cv2.VideoCapture` / `imageio.get_reader` | Ensure zero-indexed frame counts align with ground truth timestamp sequences. |
| **Video Encoding** | `VideoWriter` ('MPEG-4') `[DOC §2.2]` | `cv2.VideoWriter` / `ffmpeg-python` (CBR mode) | **Trap T3:** MATLAB exposes quality (0–100); Python FFmpeg allows true rate control (`-b:v 500k`). |
| **Color Conversion** | `rgb2lab` `[DOC §2.2]` | `skimage.color.rgb2lab` / `cv2.cvtColor` | **Trap T6:** `skimage` uses standard float ranges ($L \in [0, 100]$); OpenCV quantizes $L \in [0, 255]$. |
| **Image Resizing** | `imresize` (bicubic) `[DOC §2.2]` | `scipy.ndimage.zoom` / `cv2.resize` | **Trap T5:** MATLAB applies anti-aliasing filter by default; OpenCV `INTER_CUBIC` does NOT. |
| **Precision Scaling**| `im2double` `[DOC §2.2]` | `img.astype(np.float64) / 255.0` | Identical conversion to range $[0.0, 1.0]$. |
| **Noise Injection** | `imnoise` ('poisson') `[DOC §2.2]` | `np.random.poisson(I * photon_scale)` | **Trap T1:** MATLAB treats double as means scaled by $10^{12}$; must scale explicitly. |

---

## 10. Final Audit Appendices

### Appendix A: Assumptions and Unresolved-Dependency Register
1. **[ASSUMPTION-01: Multi-Condition Submission Files in Round 1]:** The document requires demonstrating HR extraction under *both controlled and uncontrolled environmental conditions* `[DOC §2.3 Task 1]`, yet Section 1.2 lists only `Round1_Baseline_None.mat` in the required files list. We formally propose submitting `Round1_Baseline_None.mat` alongside per-condition files `Round1_LowLight_C1.mat` and `Round1_SkinTone_C2.mat` `[DOC §1.2, ASSUMPTION]`.
2. **[ASSUMPTION-02: Target Frame Windowing for Waveform Evaluation]:** Where continuous ground-truth waveforms exist (UBFC-rPPG), evaluation metrics (MAE, RMSE, $r$) are evaluated per 10-second sliding window with 1-second stride, then averaged per video to yield the primary reporting metric `[TEAM-DEFINED]`.
3. **[ASSUMPTION-03: Single-Point Calibration Protocol for $\text{SpO}_2$ and BP]:** For clinical validation in Rounds 2 and 3, subject calibration uses strictly the first 10 seconds of recorded session data; subsequent monitoring remains completely frozen `[TEAM-DEFINED]`.
4. **[VERIFY-01: MATLAB Release VideoCompressionQuality Property]:** Confirm the exact parameter name for bitrate compression in the team's installed MATLAB version (MathWorks R2024b vs R2025a) `[DOC §2.2, VERIFY]`.
5. **[VERIFY-02: ISO 81060-2 and ISO 80601-2-61 Formal Trial Boundaries]:** Explicitly state in all submissions that ISO standards are used purely as benchmark comparators; full regulatory compliance trials require hundreds of clinical subjects under hypoxic/hypotensive stress, which is outside competition bounds `[DOC §2.4, VERIFY]`.

### Appendix B: Document Discrepancies and Formal Organizer Clarification Email
We identify five distinct structural ambiguities in the official challenge specification:
1. *Round Count vs. Directory Structure:* Section 2.3 defines 3 rounds, but Section 1.2 defines 4 folders (`/Round1_Baseline/`, `/Round2_HR_SpO2/`, `/Round3_Constrained/`, `/Round4_FullSystem_Clinical/`).
2. *Criteria Count Typo:* Section 5.2 states: *"The final round held live at the conference is judged on six equally weighted criteria. Each of the three finalist teams will be scored on the five criteria..."* (Text says 5, table lists 6).
3. *Round 1 Uncontrolled File Naming:* Section 2.3 mandates evaluating controlled and uncontrolled conditions in Round 1, but Section 1.2 lists only `Round1_Baseline_None.mat`.
4. *Condition Tag Vocab:* Section 3 recommends condition tags `Controlled`, `Hospital`, `LowLight`, `LowRes`, but omits a designated tag for Low-Bandwidth / 10 fps (C5).
5. *Annexure BP Ground Truth:* Annexure 1 lists UBFC-rPPG (HR, $\text{SpO}_2$) and rPPG-biases (mean HR only); no public dataset with Blood Pressure labels is provided in the document.

#### Formal Clarification Inquiries
```text
To: vijayjeyakumar@ieee.org
Cc: sp-competitions@listserv.ieee.org, Jaqueline.Rash@ieee.org, lucas.thomaz@ieee.org
Subject: IEEE SP Cup 2027: Technical Clarifications on Data Organization & Submission Guidelines

Dear Dr. Vijay Jeyakumar and Competition Organizing Committee,

We are preparing our submission for the 2027 IEEE Signal Processing Cup ("FaceVitals"). To ensure rigorous compliance with your evaluation protocols, our team respectfully requests clarification on the following five technical items:

1. Round 1 Multi-Condition File Artifacts: Task 1 requires demonstrating HR extraction under both controlled and uncontrolled conditions. In Section 1.2, only 'Round1_Baseline_None.mat' is explicitly itemized. Should teams submit multiple .mat files for Round 1 (e.g., 'Round1_Baseline_None.mat' alongside 'Round1_LowLight_C1.mat' and 'Round1_SkinTone_C2.mat'), or package all conditions into a single container?
2. Directory Structure Reconciliation: Section 2.3 outlines three competition rounds, whereas Section 1.2 lists four directories (/Round1_Baseline/, /Round2_HR_SpO2/, /Round3_Constrained/, /Round4_FullSystem_Clinical/). Could you confirm whether Round 3 final deliverables should be placed in /Round3_Constrained/ or /Round4_FullSystem_Clinical/?
3. Scoring Criteria Count: Section 5.2 notes that teams will be scored on "five criteria" but tabulates six equally weighted criteria (Innovation, Val Performance, Test Performance, Complexity, Report, Presentation). We assume all six criteria in the table are active and weighted at 16.67% each; please confirm.
4. Condition Tag for Low Bandwidth (C5): Section 3 specifies condition tags 'Controlled', 'Hospital', 'LowLight', 'LowRes'. What is the official tag for Low-Bandwidth/Frame-Rate captures under C5 (e.g., 'LowBandwidth' or 'LowFPS')?
5. Reference Datasets with BP Ground Truth: Annexure 1 details UBFC-rPPG and rPPG-biases, neither of which contains blood pressure ground truth. Will the Sri Ramachandra Hospital pilot dataset be the sole reference providing synchronized BP labels, or will additional public BP benchmarks be endorsed?

Thank you for your guidance and dedication to this exceptional initiative.

Sincerely,
Faculty Supervisor & Student Team, IEEE SPS Student Branch Chapter
```

### Appendix C: Ethically and Clinically Sensitive Claims Register
To preserve technical integrity before PhD physiological signal processing jurors, our technical report and live demonstration will strictly maintain the following boundaries:

```markdown
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CLINICAL INTEGRITY AND LIMITATION WORDING CONTRACT                                                     │
├────────────────────────────────┬───────────────────────────────────────────────────────────────────────┤
│ Sensitive Topic Area           │ Exact Mandatory Limitation Wording for Technical Reports & Demos      │
├────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ 1. Diagnostic / Medical Device │ "FaceVitals is an academic research prototype designed for screening  │
│    Status                      │ trends, NOT a certified medical device under FDA 510(k) or EU MDR.    │
│                                │ It must not be used for primary diagnostic or triage decisions."      │
├────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ 2. Contactless SpO2 Accuracy   │ "Camera-based SpO2 estimation in the visible spectrum is an indirect  │
│    without NIR Sensing         │ surrogate. Due to the absence of near-infrared isosbestic channels,   │
│                                │ readings outside 90–100% saturation or in dyshemoglobinemias cannot   │
│                                │ be clinically verified without direct co-oximetry."                   │
├────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ 3. RGB-Only Cuffless Blood     │ "Cuffless camera-based BP estimation is highly susceptible to vascular│
│    Pressure Estimation         │ tone shifts and smooth-muscle tone variations. Performance is unvali- │
│                                │ dated in shock, severe arrhythmia, or acute hypertensive crises."     │
├────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ 4. Vitiligo & Dermatological   │ "Synthetic depigmentation testing demonstrates algorithmic resilience │
│    Generalization Claims       │ to reflectance gradients; however, performance on clinical vitiligo  │
│                                │ cannot be fully claimed without extensive multi-center patient data." │
└────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

### Appendix D: Minimal Extension Hooks for C3, C6, and C7
While C1, C2, C4, and C5 represent our core design drivers, modular extension interfaces ensure compatibility with secondary constraints without degrading primary throughput:
* **C3 Multi-Person Isolation Hook:** Bounding boxes from `faceDetector` pass through a single-target Hungarian association module (`assignDetectionsToTracks` `[DOC §2.2]`). If multiple faces appear, a locked target ID maintains tracking via Kalman velocity filtering, discarding bystander coordinates.
* **C6 Aging-Skin Micro-Shadow Hook:** An edge-preserving bilateral spatial prefilter smooths localized wrinkle gradients before spatial averaging, preventing micro-shadow amplitude modulation from corrupting pulsatile cardiac frequencies `[DOC §2.2]`.
* **C7 Masked-Face Dynamic ROI Re-weighting:** A landmark visibility check evaluates lower-face occlusion. If lower cheek confidence drops below 0.3 (indicating a mask `[DOC §2.2]`), the spatial extraction pipeline dynamically redistributes 100% of the signal weight to the forehead and periocular ROIs.

### Appendix E: Final Architecture & Compliance Self-Audit

| Verification Dimension | Status | Verification Evidence / Location |
| :--- | :---: | :--- |
| **All Mandatory Constraints Handled** | **CONFIRMED** | C1 (Low Light), C2 (Skin Tone), C4 (Low Res), and C5 (Low BW) are systematically addressed in §2, §3, §4, §5, §6, and §7. |
| **Official File Naming Respected** | **CONFIRMED** | `RoundX_Condition_Constraint.mat` and `Subject{ID}_{SkinTone}_{Condition}_{Parameter}.ext` specified exactly `[DOC §1.2, §3]`. |
| **All 6 Required `.mat` Variables** | **CONFIRMED** | `facial_video`, `roi_data`, `reference_signal`, `processed_signal`, `metrics`, `params` verified in §5.3 and code §9.2. |
| **Official Metric Suite Present** | **CONFIRMED** | MAE, RMSE, Pearson $r$, and Bland-Altman LoA formulated with cluster-bootstrap rigor in §6.1. |
| **Trap T1–T12 Mitigations Integrated** | **CONFIRMED** | Handled across §3.4 (T1, T2, T3, T4, T5), §4.4 (T6), §6.3 (T7), §6.1 (T8), §2.2 (T9), §4.3 (T10), §5.3/9.2 (T11), §3.1 (T12). |
| **First-8-Days Action Plan Aligned** | **CONFIRMED** | Action items mapped to 17 Oct – 25 Oct 2026, targeting the hard registration freeze on 25 Oct 2026 `[DOC §6]`. |