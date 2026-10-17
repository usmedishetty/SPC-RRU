# FaceVitals: Implementation-Ready Technical Blueprint and Competition Execution Plan (v2)
**2027 IEEE Signal Processing Cup**  
*Theme: Equitable, Edge-Deployable Camera-Based Vital Sign Monitoring*  
**Planning Date:** 17 October 2026 | **Target Conference:** IEEE ICASSP 2027 (16–21 May 2027, New Orleans, USA)  
**Document Classification:** Technical Architecture & Operational Plan (Revision 2.0, Post-Editorial Audit)

---

## 1. Executive Summary, Thesis and Scoring Map

### 1.1 Core Engineering Thesis
> An equitable, edge-deployable camera-based vital signs platform cannot rely on black-box end-to-end deep learning or uncalibrated classical rPPG under extreme real-world stress. We present an **SQI-Conditioned Hybrid Physiological Architecture**: a dual-path framework fusing a deterministic, physics-grounded chrominance path (Plane-Orthogonal-to-Skin / CHROM) with a lightweight 3D-separable spatiotemporal convolutional neural network (PhysNet-Lite, target ~0.85M parameters / measured: [MEASURE]). Three specialized Signal Quality Indices (${\text{SQI}}_{\text{fuse}}$, ${\text{SQI}}_{\text{win}}$, ${\text{SQI}}_{\text{abstain}}$) dynamically govern classical-deep fusion arbitration, temporal window selection (evaluated at a fixed 10.0 s wall-clock window across all frame rates, with an adaptive windowing sensitivity experiment spanning 3.0 s to 15.0 s), and calibrated abstention. By training the neural trunk via stratified loss reweighting / Group-DRO across six Individual Typology Angle strata (ITA-1 to ITA-6) and validating through an auto-exposure-aware, linear-light degradation simulator on a physical validation cohort, the platform prevents silent catastrophic failure under low light (<= 50 lux, C1), diverse skin pigmentation (C2), spatial decimation (<= 320x240, C4), and low bandwidth (<= 500 kbps, <= 10 fps, C5 via Mode A trace streaming and Mode B compressed video) [TEAM-DEFINED].

### 1.2 Context, Related Work & Selection of Headline Innovations

#### Related Work & Background
Camera-based remote photoplethysmography (rPPG) extracts pulsatile blood volume changes from subtle facial color variations. Despite major advances, four fundamental vulnerabilities persist in competitive and clinical benchmarks:
1. **Skin-Tone Disparity:** Melanin in the epidermis acts as an optical attenuation filter, attenuating short visible wavelengths (blue/green) much more severely than longer red wavelengths. Studies by Nowara et al. (2020) and Dasari et al. (2021) demonstrated that standard rPPG algorithms exhibit up to 3x higher error on dark skin tones (Fitzpatrick V–VI) due to attenuated AC pulsatile amplitude. The AiPEX-Lab `rppg_biases` benchmark was established specifically to highlight this systemic bias [DOC §10].
2. **Deep Neural Network Vulnerability:** While deep models (PhysNet, DeepPhys, TS-CAN) achieve low error under pristine laboratory conditions, they overfit to unconstrained illumination shifts and high-frequency sensor noise, often failing catastrophically under severe low-light conditions (<= 50 lux).
3. **Bandwidth and Frame-Rate Constraints:** Real-world telehealth streaming from remote rural clinics frequently drops below 10 fps and 500 kbps, introducing severe temporal aliasing and MPEG-4 block compression artifacts.
4. **Physiological Boundaries of RGB BP & SpO2:** Photoplethysmography measures volumetric blood changes, not pressure directly. Pulse Transit Time (PTT) requires resolvable time differences between two arterial sites. Sub-frame delays can be estimated in principle by phase or cross-correlation, but are not reliably resolvable at these low SNRs and frame rates (especially <= 10 fps). Therefore, facial rPPG Blood Pressure estimation must rely strictly on pulse wave morphology and verified non-demographic vascular features [TEAM-DEFINED].

```text
                  ┌─────────────────────────────────────────────────────────┐
                  │ Innovation Screening: Defensibility & Falsifiability    │
                  └─────────────────────────────────────────────────────────┘
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
    [ACCEPTED CANDIDATES]                                               [REJECTED CANDIDATES]
    1. Multi-Scale SQI-Conditioned Hybrid Stacking                       A. Super-Resolution Pre-Stage (ESRGAN/SRCNN)
       - Falsifiable: Ablating SQI gating increases                        - Rejection: Multiplies FLOPs >12x, hallucinates
         worst-stratum error by >= 20% [PROVISIONAL].                        high-frequency PPG artifacts in dim light.
    2. Linear-Light Simulator & Real-World Validation                    B. Facial PTT-Based Multi-Site Blood Pressure
       - Falsifiable: Linear-light simulator narrows                       - Rejection: Sub-frame delays are not reliably
         domain gap on real 50-lux clips by >= 2 bpm [PROVISIONAL].          resolvable at low SNRs and frame rates (<= 10 fps).
    3. Stratified Minimax Optimization over ITA Strata                   C. Blind End-to-End Black-Box Video Transformer
       - Falsifiable: Minimax optimization bounds                          - Rejection: OOM on Raspberry Pi 4/5, uninterpretable
         inter-stratum disparity to |Delta| <= 1.5 bpm [PROVISIONAL].        failure modes under sensor shot noise.
```

#### The Three Selected Headline Innovations [TEAM-DEFINED]
1. **Innovation 1: Multi-Scale SQI-Conditioned Classical-Deep Stacking**
   * *Incremental Gap Filled:* Existing hybrid rPPG architectures either use fixed heuristic averaging or hand-tuned gating thresholds (beta, theta_0). We implement a validation-trained logistic stacking arbiter conditioned on multi-channel signal quality indices (${\text{SQI}}_{\text{fuse}}$), dynamically weighting physics-grounded chrominance projections (${\mathbf{S}}_{\text{POS}}$) and deep spatial representations (${\hat{y}}_{\text{deep}}$).
   * *Falsifiable Hypothesis:* Dynamic ${\text{SQI}}_{\text{fuse}}$ stacking yields lower worst-case MAE than either branch alone under combined stress (<= 50 lux, QVGA, 10 fps).
   * *Ablation to Confirm/Kill:* Compare against static averaging (alpha = 0.5) and isolated single branches (alpha in {0, 1}). If dynamic stacking does not reduce worst-group MAE by >= 20% [PROVISIONAL: target 20% / measured: [MEASURE]], reject.
2. **Innovation 2: Linear-Light, Auto-Exposure-Aware Degradation Simulator Validated on Physical Clips**
   * *Incremental Gap Filled:* Standard rPPG synthetic degradation naively attenuates sRGB pixel values, ignoring camera gamma curves, photon Poisson statistics, sensor read noise, and auto-exposure digital gain amplification. We formulate a linear irradiance simulation pipeline with dual reporting (dark-clip vs. brightness-normalized-noisy) and calibrate it directly against physical lux-meter recordings.
   * *Falsifiable Hypothesis:* Training on linear-light Poisson-Gaussian simulated data with digital gain normalization reduces the cross-domain generalization error on real physical low-light clips (<= 50 lux) compared to naive sRGB scaling.
   * *Ablation to Confirm/Kill:* Train on naive sRGB scaling vs. linear-light simulator v2; evaluate on 15 real physical low-light clips. If generalization MAE does not improve by >= 2.0 bpm [PROVISIONAL: target 2.0 bpm / measured: [MEASURE]], reject.
3. **Innovation 3: Stratified Minimax Optimization over ITA Strata**
   * *Incremental Gap Filled:* Standard Empirical Risk Minimization (ERM) minimizes cohort-average loss, sacrificing darker skin tones where pulsatile SNR is naturally lower. We discretize skin reflectance into six objective Individual Typology Angle bins (ITA-1 to ITA-6) and optimize via stratified loss reweighting / Group-DRO.
   * *Falsifiable Hypothesis:* Minimax group optimization bounds the maximum per-stratum deviation from population mean error across all six ITA bins to within tolerance tau.
   * *Ablation to Confirm/Kill:* Compare standard ERM vs. Stratified Reweighting vs. Group-DRO. If stratified minimax fails to bound inter-stratum deviation to |Delta| <= 1.5 bpm [PROVISIONAL: target 1.5 bpm / measured: [MEASURE]] on validation folds with n_g >= 15, or increases mean cohort error by > 10% [PROVISIONAL], reject.

### 1.3 Official Competition Round Progression
* **Round 1 (Baseline & Initial Constraints):** Controlled and uncontrolled HR extraction from RGB video [DOC §2.3 Task 1]. Core pipeline relies on classical CHROM/POS, unified SQI, physics simulators, and full statistical evaluation. Deep model serves as a pretrained reference / ablation [TEAM-DEFINED]. Evaluates controlled baseline and >= 2 constraints (C1 Low-Light and C4 Low-Resolution implemented deeply; C2 evaluated across public strata; C5 simulation sweep) [DOC §2.3 Task 1]. *Submission: 30 Nov 2026; Results: 05 Dec 2026 [DOC §6].*
* **Round 2 (Two-Parameter Joint Estimation):** Extend pipeline to jointly estimate Heart Rate and SpO2 without contact/thermal sensing [DOC §2.3 Round 2]. Quantitative validation against clinical pulse oximetry across full multi-constraint matrix [DOC §2.3 Round 2]. *Submission: 16 Jan 2027; Results: 21 Jan 2027 [DOC §6].*
* **Round 3 (Full Three-Parameter Clinical & Edge System):** Simultaneous estimation of HR, SpO2, and Blood Pressure (Systolic/Diastolic) [DOC §2.3 Round 3]. Mandatory clinical validation against reference datasets and clinical devices [DOC §2.3 Round 3]. Real-time live demonstration on an edge device (Raspberry Pi target) under >= 2 constraints at IEEE ICASSP 2027 in New Orleans, USA (16–21 May 2027) [DOC §2.3 Round 3, §6].

### 1.4 Scoring Map: Judging Criteria to Blueprint Implementation

| Evaluation Criterion [DOC §5.2] | Weight [DOC §5.2] | Round 1 Mapping [DOC §2.3 Task 1] | Final Round Blueprint Element | Evidence Artifact Produced |
| :--- | :--- | :--- | :--- | :--- |
| **1. Innovation & Thoroughness** | 16.67% (1/6) | Primary (Innovation) | §1.2: 3 falsifiable hypotheses; §3.4: OFAT sweeps + combined stress cells; §4.1: Ablation ladder | Ablation matrix table; synthetic-to-real domain gap curves; stacking logs |
| **2. Validation-Set Performance** | 16.67% (1/6) | Primary (Feasibility) | §3.2: 5-fold subject-disjoint CV on dev cohort; §6.1-6.2: Equivalence testing; §4.2: Fair loss balance | Stratified CV tables with 95% bootstrap CIs; risk-coverage curves |
| **3. Test-Set Performance** | 16.67% (1/6) | Primary (Feasibility) | §3.2: Blind held-out subject cohort; §6.1-6.3: MAE, RMSE, r, Bland-Altman; C2 TOST test | Test results CSV; Bland-Altman agreement plots; ITA disparity histograms |
| **4. Model & Time Complexity** | 16.67% (1/6) | Secondary (Feasibility)| §2.5: Edge profiling spike; ONNX/ARM runtime; INT8 QAT benchmark; Mode A trace streaming | Latency profiles (ms/frame), memory footprints (MB), edge FPS logs |
| **5. Quality & Clarity of Report** | 16.67% (1/6) | Primary (Documentation)| §5.1: Strict IEEE format [ASSUMPTION: 4-page R1 / 6-page Final]; full traceability tags [DOC] | Complete IEEE-format PDF with vector graphics and explicit limitations |
| **6. Quality & Clarity of Presentation**| 16.67% (1/6) | N/A (Round 3 only) | §8.1: Live interactive GUI demo architecture on Raspberry Pi 4/5; failsafe backup video logs | Slide deck, recorded video demonstration, live hardware test harness |

### 1.5 Mandatory vs. Optional Competition Scope Matrix

| Dimension | Mandatory Specification [DOC] | Team Strategy & Selected Stance | Traceability |
| :--- | :--- | :--- | :--- |
| **Primary Parameters** | HR (R1), HR+SpO2 (R2), HR+SpO2+BP (R3) | Dual-path SQI-conditioned multi-task model | [DOC §2.1, §2.3] |
| **Sensing Modality** | Standard RGB video only (no NIR, thermal, contact) | 3-channel visible spectrum rPPG (Forehead/Cheeks) | [DOC §2.1] |
| **Primary Constraints** | >= 2 constraints in R1 & R3; C1, C2, C4, C5 priority | Deep focus: C1 (Light), C4 (Res), C2 (Skin), C5 (BW) | [DOC §2.2, §2.3] |
| **C5 Operating Modes** | On-device or low-bandwidth offline mode | Mode A (ROI Trace Stream, ~4.3 kbps) & Mode B (Video) | [DOC §2.2 C5, TEAM-DEFINED] |
| **Secondary Constraints**| C3 (Multi-person), C6 (Aging), C7 (Masks) | Minimal modular extension hooks (Appendix D) | [DOC §2.2] |
| **Target Platform** | Raspberry Pi is optional / under evaluation | Raspberry Pi 4/5 optimized (ONNX ARM64 / C++ engine) | [DOC §2.1, §5.2] |
| **Implementation Language**| MATLAB or Python | Python (PyTorch/ONNX) training/edge + MATLAB submissions | [DOC §2.3 Task 1, §7.2] |

---

## 2. End-to-End System Architecture

### 2.1 End-to-End System Block Diagram

```text
========================================================================================================================
                                     FACEVITALS END-TO-END PIPELINE ARCHITECTURE (v2)
========================================================================================================================

  [STAGE 1: ACQUISITION & C5 OPERATING MODES]
  RGB Stream / Offline MP4 ──► Session Metadata Logging (fps, lux, res, distance, nominal & achieved kbps)
                                │
                                ├──► [Mode A: Edge Trace Extraction] ──► 3 ROIs x 3 ch @ 30 Hz (4.32 kbps) ──► Server
                                └──► [Mode B: Degraded Video Stream] ──► Sub-sampled / MPEG-4 (<= 500 kbps, <= 10 fps)
                                │
                                ▼
  [STAGE 2: PREPROCESSING & QUALITY GATING]
  Face Detection & Landmarks ──► Forehead & Bilateral Cheek Extraction ──► Illumination & Noise Gating
  (MediaPipe / Haar Fallback)   (Mesh Bounding Boxes with Fallback)        (Photon Noise & Auto-Exposure Gain Check)
                                │                                        │
                                ├────────────────────────────────────────┴──────────────┐
                                ▼                                                       ▼
  [STAGE 3A: CLASSICAL rPPG PATH]                                         [STAGE 3B: DEEP SPATIOTEMPORAL PATH]
  Spatial Averaging per ROI c(t)                                          Normalized Tensor X: [B x 3 x T x 64 x 64]
  Linear Normalization c_n(t) = c(t)/mu_c - 1                             PhysNet-Lite (3D-Separable ResNet Trunk)
  Projection: POS & CHROM Waveform Generators                             ├── T-length Waveform Decoder (BVP pulse)
  Causal 4th-order IIR Bandpass (0.7-3.0 Hz)                              └── Globally Pooled Latent Vector z [B x 128]
                                │                                                       │
                                └────────────────────────────────────────┬──────────────┘
                                                                         ▼
  [STAGE 4: TRI-SQI ARBITRATION & MULTI-TASK ESTIMATION]
  Tri-SQI Engine:
  ├── SQI_win: Governs wall-clock window length (10.0 s primary; adaptive 3-15 s experiment)
  ├── SQI_fuse: Governs logistic stacking gate between POS and PhysNet-Lite spectral peaks
  └── SQI_abstain: Calibrated risk-coverage gate rejecting unresolvable windows
                                │
                                ├───────────────────┬───────────────────┐
                                ▼                   ▼                   ▼
                          [HR Head]           [SpO2 Head]          [BP Head]
                          Spectral Product    Ablation: R/G vs    Nonlinear Pulse Morphology
                          Hamming PSD Peak    R/B vs Learned      + b_demo(Age, BMI)
                          Parabolic Refine    Stacked Latent Reg  [NO ITA IN BP INPUT]
                          Output: BPM         Output: % Sat       Output: SBP/DBP (mmHg)
                                │                   │                   │
                                └───────────────────┼───────────────────┘
                                                    ▼
  [STAGE 5: CALIBRATED UNCERTAINTY & CONFORMAL ABSTENTION]
  Heteroscedastic Uncertainty + Conformal Bounds ──► If sigma_pred > theta_abstain OR SQI_abstain < 0.40 ──► ABSTAIN (NaN)
                                                 ──► Else Output Validated Vitals + 95% Conformal Coverage Bounds
========================================================================================================================
```

### 2.2 Window Length, Frequency Resolution & Constraint Matrix

#### Fundamental Trade-Off: Window Duration vs. Frequency Resolution
A fundamental law of Fourier analysis states that spectral resolution depends strictly on the window observation duration $T$ in seconds:

```math
\Delta f = \frac{1}{T} \text{ Hz} \implies \Delta \text{HR} = \frac{60}{T} \text{ bpm}
```

Expanding window length from 3 s to 12 s improves frequency resolution from 20 bpm to 5 bpm; it does **not** "preserve" $\Delta f$. Furthermore, keeping a constant frame count across varying frame rates changes the physical observation time, severely corrupting resolution.

| Window Length $T$ (s) | Sampling Rate $f_s$ | Frame Count $N$ | Spectral Resolution $\Delta f$ | HR Bin Width $\Delta \text{HR}$ | Latency to First Estimate | Pipeline Update Interval |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0 s** | 30 fps | 90 | 0.333 Hz | 20.0 bpm | 3.0 s | 1.0 s |
| **5.0 s** | 30 fps | 150 | 0.200 Hz | 12.0 bpm | 5.0 s | 1.0 s |
| **10.0 s (Primary)** | **30 fps** | **300** | **0.100 Hz** | **6.0 bpm** | **10.0 s** | **1.0 s** |
| **10.0 s (Primary)** | **10 fps** | **100** | **0.100 Hz** | **6.0 bpm** | **10.0 s** | **1.0 s** |
| **12.0 s** | 10 fps | 120 | 0.083 Hz | 5.0 bpm | 12.0 s | 1.0 s |
| **15.0 s (Stress)** | 10 fps | 150 | 0.067 Hz | 4.0 bpm | 15.0 s | 1.0 s |
| **30.0 s** | 10 fps | 300 | 0.033 Hz | 2.0 bpm | 30.0 s | 2.0 s |

*Evaluation Protocol Decision [TEAM-DEFINED]:* For all C5 comparisons across frame rates (30, 15, 10, 7.5, 6 fps), the primary benchmark uses an **identical 10.0 s wall-clock window** ($\Delta \text{HR} = 6.0\,\text{bpm}$, refined via parabolic interpolation). Adaptive windowing ($3.0\text{--}15.0\,\text{s}$) is evaluated strictly as an isolated, labeled secondary experiment.

#### Architectural Constraint Handling Matrix

| Pipeline Stage | Input $\to$ Output Shape | Method & Algorithmic Justification | Handling C1 (Low Light $\le 50\,\text{lux}$) | Handling C2 (Skin Tone / Vitiligo) | Handling C4 (Low Res $\le 320\times 240$) | Handling C5 (Low BW $\le 10\,\text{fps}$, $\le 500\,\text{kbps}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Acquisition & Mode Selection** | Bitstream $\to$ $[T \times H \times W \times 3]$ uint8 | Dynamic stream ingestion; hardware timestamp logging [DOC §1.2]. | Measures physical lux from sensor exposure + histogram; checks AGC gain. | Logs metadata ITA stratum tag (ITA-1 to ITA-6) and Von Luschan tile reading. | Ingests native resolution without unverified super-resolution hallucination. | Mode A: extracts ROI traces on-device (4.32 kbps). Mode B: CBR MPEG-4 stream. |
| **2. Face & Landmark Tracking** | $[H \times W \times 3] \to$ BBox $[4]$, Mesh $[468 \times 2]$ | MediaPipe FaceMesh / Haar Cascade fallback + KLT tracking [DOC §2.2]. | Adaptive Y-channel histogram stretch ensures landmark convergence. | Geometry-based facial topology; invariant to epidermal melanin concentration. | Sub-pixel tracking; auto-switches to face bounding box if landmark jitter > 5 px. | Runs landmark detection every $K=5$ frames; tracks via lightweight KLT in-between. |
| **3. Multi-ROI Extraction** | Frame + Mesh $\to$ 3 ROIs: $[3 \times T \times H_r \times W_r \times 3]$ | Segment Forehead ($60\times 30$), Bilateral Cheeks ($25\times 25$) [DOC §10]. | Convex-hull trimming avoids cast shadows on facial periphery. | Sub-ROI Mahalanobis anomaly detection; SNR-weighted fusion masks vitiligo patches. | Dynamic ROI scaling down to minimum $12\times 12$ px spatial floor. | Decouples spatial cropping from transmission in Mode A. |
| **4A. Classical Projection** | $[3 \times T] \to [T]$ temporal rPPG signal | POS & CHROM projection + causal 4th-order Butterworth IIR [DOC §2.1, §2.3]. | Eliminates common-mode intensity fluctuations from dim sensor noise. | Standardizes skin reflectance plane across wide range of melanin absorption. | Spatial pixel averaging over ROI acts as low-pass filter, recovering SNR at QVGA. | Causal IIR filtering with group delay compensation; avoids forward-backward filter lag. |
| **4B. Deep Spatiotemporal Branch** | $[B \times 3 \times T \times 64 \times 64] \to [B \times T], [B \times 128]$ | PhysNet-Lite: (2+1)D factorized separable convs + waveform decoder head. | Group normalization and channel attention layers capture faint periodic gradients. | Stratified reweighting / Group-DRO prevents feature suppression on dark skin. | Input spatial pooling preserves phase-coherent color shifts despite decimation. | Evaluates FiLM conditioning and temporal dilation scaling for low-fps streaming. |
| **5. Tri-SQI Arbitration** | Signals $\to {\text{SQI}}_{\text{fuse}}, {\text{SQI}}_{\text{win}}, {\text{SQI}}_{\text{abstain}}$ | Multi-channel entropy, peak SNR, and optical flow tracking variance. | Penalizes elevated sensor read-noise floor and digital-gain noise boost. | Weights bilateral cheeks independently if localized vitiligo alters one side. | Penalizes landmark drift and motion blur common in low-resolution streams. | Flags spectral broadening and harmonic ambiguity caused by 10 fps decimation. |
| **6. Multi-Task Heads (HR/SpO2/BP)** | Features $\to$ HR, SpO2, SBP, DBP | Multi-task heads with heteroscedastic uncertainty estimation [TEAM-DEFINED]. | Historical Kalman tracking prevents frequency lock on sensor noise. | Stratified calibration offsets per ITA cluster; eliminates bias drift. | Resolution-invariant latent embedding preserves vital calibration baselines. | Evaluated at fixed 10.0 s window; harmonic checks prevent mains-flicker locking. |
| **7. Conformal Abstention** | Vitals + $\sigma^2 \to$ Validated Vitals or $\text{NaN}$ | Mondrian (per-stratum) conformal prediction intervals and risk-coverage gate. | Rejects windows where ambient illuminance drops below shot-noise floor. | Flags uncalibrated vitiligo edge cases rather than outputting erroneous readings. | Abstains if face area falls below $32\times 32$ pixels in total frame. | Rejects windows affected by burst packet drop or temporal frame skips > 200 ms. |

### 2.3 Classical vs. Deep Dual-Path Design Rationale
* **The Classical Branch (POS/CHROM):** Provides an instantaneous, zero-training baseline that cannot overfit. Plane-Orthogonal-to-Skin (POS) projects normalized RGB signals onto two orthogonal planes:

```math
S_1(t) = R_n(t) - G_n(t), \quad S_2(t) = R_n(t) + G_n(t) - 2B_n(t)
```

```math
\text{POS}(t) = S_1(t) + \frac{\sigma(S_1)}{\sigma(S_2)} S_2(t)
```

* **The Deep Spatiotemporal Branch (PhysNet-Lite):** Employs factorized (2+1)D separable convolutions, outputting a continuous $T$-length blood volume pulse waveform plus a globally pooled latent vector $\mathbf{z} \in \mathbb{R}^{128}$ [TEAM-DEFINED].
* **The Tri-SQI Arbitration Mechanism:** The system computes three dedicated quality indices:
  1. ${\text{SQI}}_{\text{fuse}} \in [0, 1]$: Governs dynamic stacking between classical POS and deep spectral distributions via a validation-learned logistic gate.
  2. ${\text{SQI}}_{\text{win}} \in [0, 1]$: Regulates adaptive temporal expansion ($3\text{--}15\,\text{s}$) in secondary streaming experiments.
  3. ${\text{SQI}}_{\text{abstain}} \in [0, 1]$: Flags unresolvable physiological noise floors, triggering calibrated abstention.
* **Conditioning on Operational Parameters:** Rather than training brittle disjoint models for every permutation of lighting and resolution, the deep trunk evaluates Feature-wise Linear Modulation (FiLM) conditioning on normalized metadata vectors $[\text{lux}_{\text{norm}}, \text{res}_{\text{norm}}, \text{fps}_{\text{norm}}]$ vs. temporal dilation scaling [TEAM-DEFINED].

### 2.4 Multi-Task Parameter Heads & Signal Processing Formulation
1. **Heart Rate (HR):**
   * Computed from cross-spectral density / product of normalized PSDs between classical POS and deep waveforms:

```math
P_{\text{fused}}(f) = P_{\text{POS}}(f)^{\left(1 - \alpha\right)} \cdot P_{\text{deep}}(f)^{\alpha}, \quad \alpha = \sigma\left(\mathbf{w}^T [{\text{SQI}}_{\text{fuse}}, \text{SNR}_{\text{POS}}, \text{SNR}_{\text{deep}}] + b\right)
```

```math
f_{\text{pulse}} = \arg\max_{f \in [0.7, 3.0]\,\text{Hz}} P_{\text{fused}}(f), \quad \text{HR} = 60 \times f_{\text{pulse}} \text{ bpm}
```

   * Refined via parabolic peak interpolation and verified against an explicit harmonic ratio check ($2 \times f_{\text{pulse}}$) [DOC §2.3 Task 1].
2. **Blood Oxygen Saturation (SpO2):**
   * *Actual Physical Limits of RGB Optical Sensing:* Unlike clinical pulse oximeters with narrow monochromatic LED bands at $660\,\text{nm}$ and $940\,\text{nm}$, visible RGB cameras suffer from:
     - Broad overlapping spectral sensitivities with significant channel cross-talk.
     - Differing penetration depths: blue light ($450\,\text{nm}$) penetrates $<0.5\,\text{mm}$ (mostly epidermis), green ($530\,\text{nm}$) penetrates $\approx 1\,\text{mm}$ (capillary loops), while red ($630\,\text{nm}$) reaches deep arteriolar beds ($>2\,\text{mm}$).
     - Severe Blue-channel SNR collapse on dark skin: melanin absorption increases exponentially at shorter wavelengths, leaving the blue channel dominated by sensor read noise in low light.
     - Camera ISP non-linear tone curves, gamma correction, and dynamic Auto White Balance (AWB) corrupting absolute AC/DC ratios.
     - Codec chroma subsampling (YUV 4:2:0) and lossy quantization attenuating Red and Blue AC pulsatile amplitudes differently from Green.
   * *Formulation & Ablation:* Rather than assuming a fixed Red/Blue ratio, we systematically ablate Red/Green vs. Red/Blue vs. Learned Multi-Channel Projections:

```math
R_{\text{phys}} = \frac{\text{AC}_{\lambda_1} / \text{DC}_{\lambda_1}}{\text{AC}_{\lambda_2} / \text{DC}_{\lambda_2}}, \quad \text{SpO}_2 = \mathbf{w}_{\text{SpO2}}^T [\mathbf{z}_{\text{deep}}, R_{\text{phys}}] + b_{\text{calib}}
```

3. **Blood Pressure (BP: Systolic / Diastolic):**
   * *Physical Constraints & Demographic Integrity:* Multi-site facial Pulse Transit Time (PTT) is not reliably resolvable under these SNRs and low frame rates. Blood pressure is estimated strictly through non-linear pulse wave morphology features (systolic rise time, reflection index, crest time, normalized pulse area) combined with non-racial demographic priors:

```math
\begin{bmatrix} \text{SBP} \\ \text{DBP} \end{bmatrix} = \mathbf{W}_{\text{BP}} \phi\left(\mathbf{z}_{\text{deep}}, \text{waveform morphology}\right) + \mathbf{b}_{\text{demo}}\left(\text{Age}, \text{BMI}\right)
```

   * *Ethical Boundary:* Individual Typology Angle (ITA) is strictly excluded from $\mathbf{b}_{\text{demo}}$ (A6). Any demographic-only fallback is explicitly labeled "non-video-derived baseline". Benchmarked against ISO 81060-2 (mean error <= 5 mmHg, SD <= 8 mmHg) as an aspirational reference [DOC §2.4, VERIFY]. Pretrained on contact PPG (PulseDB) with an explicit non-contact domain-gap caveat [TEAM-DEFINED].

### 2.5 Edge Deployment Targets & Budget Specification [TEAM-DEFINED]
* **Target Hardware:** Raspberry Pi 4 Model B (Quad Cortex-A72 @ 1.5 GHz) / Raspberry Pi 5 (Quad Cortex-A76 @ 2.4 GHz) [DOC §2.1, §5.2].
* **Complexity Scaling:** 3D-separable convolutional complexity scales linearly with window duration ($O(T)$). For a primary window of $T=10.0\,\text{s}$ ($N=300$ frames @ $30\,\text{fps}$, $N=100$ frames @ $10\,\text{fps}$), temporal convolutions are executed every 30 frames ($1.0\,\text{s}$ stride).
* **Architecture Specifications:**
  * Model Parameter Target: target ~0.85M parameters / measured: [MEASURE]. Note: PhysNet-Lite is comparable in size to original PhysNet (~0.8M) [VERIFY] (A11).
  * Computational Complexity Target: target <= 2.0 GFLOPs / measured: [MEASURE].
  * Raspberry Pi 4 Frame Latency Target: target <= 40.0 ms/frame / measured: [MEASURE].
  * Raspberry Pi 4 Throughput Target: target >= 22 fps / measured: [MEASURE].
  * Peak RAM Footprint Target: target <= 150 MB / measured: [MEASURE].
* **Edge Profiling Spike Plan (B8):** Conducted in Weeks 1–2 comparing ONNX Runtime ARM64 (NEON vectorized) vs. TFLite across FP32, FP16, and INT8. If INT8 Post-Training Quantization (PTQ) degrades rPPG SNR by > 3 dB, Quantization-Aware Training (QAT) is triggered. Standalone C++ / ONNX Runtime is selected as the primary edge deployment target; MATLAB Coder 3D-convolution compilation on ARM is evaluated under [VERIFY]. Classical-only POS pipeline serves as a zero-parameter edge fallback.

---

## 3. Data Strategy and Experiment Design

### 3.1 Data Tiering, Inventory & Critical Path Governance (B2)

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DATA TIERING, CAPABILITIES & ACCESS GOVERNANCE (B2)                                                   │
├─────────┬──────────────────────────┬───────────────────┬──────────────┬──────────────┬─────────────────┤
│ Tier    │ Dataset Name & Source    │ Parameters Logged │ Video Format │ Demographics │ Permitted Uses  │
├─────────┼──────────────────────────┼───────────────────┼──────────────┼──────────────┼─────────────────┤
│ Tier A  │ UBFC-rPPG (Annexure 1)   │ Dataset 1 (8 vid):│ Uncompressed │ Fair skin    │ Full pipeline;  │
│ (Public)│ [DOC §10]                │ HR, SpO2, PPG wave│ .avi, 30 fps │ (ITA-1 to    │ L_wave, L_spec; │
│         │                          │ Dataset 2 (42 vid)│ 640x480 VGA  │ ITA-3)       │ C1, C4, C5 sim. │
│         │                          │ HR, PPG waveform  │ [NO HD 720p!]│              │ [No 720p base]  │
├─────────┼──────────────────────────┼───────────────────┼──────────────┼──────────────┼─────────────────┤
│ Tier A  │ AiPEX-Lab rppg_biases    │ Mean HR only      │ Pre-cropped  │ India &      │ TEST ONLY for   │
│ (Public)│ (Annexure 1) [DOC §10]   │ (.csv per video). │ patches:     │ Sierra Leone │ scalar HR;      │
│         │                          │ NO waveform!      │ 60x30, 25x25 │ (ITA-4 to    │ NO face detect, │
│         │                          │ NO BP, NO SpO2!   │ [NO full face│ ITA-6)       │ NO C4 sim!      │
├─────────┼──────────────────────────┼───────────────────┼──────────────┼──────────────┼─────────────────┤
│ Tier B  │ Sri Ramachandra Hospital │ HR, SpO2, BP      │ Native RGB,  │ Diverse ITA, │ Core Clinical   │
│ (Org.)  │ Pilot Dataset (IEC/23/   │ (Synchronized     │ Hospital &   │ Elderly,     │ Benchmark;      │
│         │ FEB/177/08) [DOC §3, §8] │ clinical devices) │ Controlled   │ Clinical     │ Full multi-task │
├─────────┼──────────────────────────┼───────────────────┼──────────────┼──────────────┼─────────────────┤
│ Tier C  │ Supplementary Own IRB    │ Synchronized ECG  │ HD 1080p,    │ Diverse local│ Real 50-lux     │
│ (Local) │ Collection [DOC §3, §5.1]│ HR, PulseOx SpO2, │ 50 lux real, │ cohort       │ validation set; │
│         │                          │ Cuff BP           │ QVGA, 10 fps │ (ITA-2-ITA-6)│ HD baseline     │
└─────────┴──────────────────────────┴───────────────────┴──────────────┴──────────────┴─────────────────┤
```

*Data Contradiction Resolutions (A8):*
1. `rppg_biases` provides only video-level mean HR on pre-cropped patches ($60\times 30$ forehead, $25\times 25$ cheeks). These pre-cropped patches rule out facial landmark detection, continuous waveform loss ($\mathcal{L}_{\text{wave}}$), C4 resolution downsampling (downsampling $25\times 25$ destroys the ROI), and robust specular-free ITA estimation. It is used strictly as a held-out test set for scalar HR error across dark skin tones.
2. UBFC-rPPG is recorded at $640\times 480$ (VGA). **No public dataset in Annexure 1 provides a $1280\times 720$ (HD) baseline.** The HD baseline must originate from Tier B (organizer clinical data) or Tier C (own IRB data).
3. Access requests for external BP-labelled facial video datasets (BP4D+, V4V) are initiated on Day 2 under [VERIFY] regarding license terms.

### 3.2 Subject-Disjoint Stratified Partitioning & Held-Out Stress Sets
* **Partitioning Protocol:** 5-fold subject-disjoint cross-validation on the development cohort, with an isolated 20% blind held-out test cohort (A12).
* **Stratification Axes:** Stratified by objective ITA bins (ITA-1 to ITA-6), gender, age bands ($<30$, $30\text{--}60$, $60+$ [DOC §2.2]), and pose angles ($\pm 10^\circ, \pm 20^\circ, \pm 30^\circ$ [DOC §2.1]).
* **Minimum Sample Size Rule ($n_g \ge 15$, B7):** Any stratum with fewer than 15 subjects is reported with wide bootstrap confidence intervals and flagged as "UNDERPOWERED"; no automated pass/fail certificate is granted for that stratum.
* **Held-Out Generalization Sets:** Models trained on nominal simulated lux $\in \{50, 100, 150\}$ are evaluated on unseen stress lux $\in \{10, 20, 30\}$ and real physical low-light clips to verify domain transfer (Trap T2).

### 3.3 Ground-Truth Synchronization & Tolerances
* **Reference Modalities:** Clinical ECG / Pulse Oximeter ($\ge 100\,\text{Hz}$ for ECG) [DOC §2.3 Task 1]; calibrated oscillometric sphygmomanometer for BP [DOC §2.4].
* **Alignment Protocol:** Timestamp-aligned logging using monotonic clock hardware counters. Synchronization jitter tolerance: $\le 10.0\,\text{ms}$ [TEAM-DEFINED].

### 3.4 Simulator v2: Physics-Based Degradation Engine (A7, B5)

```text
========================================================================================================================
                          PHYSICS-BASED DEGRADATION SIMULATOR v2 (HANDLING TRAPS T1–T5, A7)
========================================================================================================================
 Baseline HD Clip (>= 300 lux, 720p, >= 30 fps)
       │
       ├──► [Order Ablation Branch A: Noise-Before-Resize] ──► Simulates low-res optical decimation of noisy field
       └──► [Order Ablation Branch B: Noise-After-Resize]  ──► Simulates native budget sensor (no artificial averaging)
       │
       ▼
 [Stage A: C1 Low-Light Linear Photon Engine]
   1. Gamma De-Quantization: I_lin = (I_sRGB / 255.0) ^ 2.2
   2. Illuminance Attenuation: I_atten = I_lin * (Lux_target / 300.0) [ASSUMPTION: 50 lux nominal gain = 0.167]
   3. Photon Conversion: N_photon = I_atten * Phi_full_well [ASSUMPTION: Phi_full_well = 5000 e-/pixel]
   4. Poisson Shot Noise: K_shot ~ Poisson(N_photon) [Trap T1: avoids MATLAB imnoise double 1e12 trap]
   5. Gaussian Read Noise: K_total = K_shot + N(0, sigma_read^2) [sigma_read = 2.5 e-]
   6. Auto-Exposure / Digital Gain Emulation (A7b):
      - Variant 1 (Dark-Clip): K_out = clip(K_total / Phi_full_well, 0, 1)
      - Variant 2 (Normalized-Noisy): K_out = clip((K_total / Phi_full_well) * (300.0 / Lux_target), 0, 1)
   7. Gamma Re-Encoding & 8-bit Quantization: I_lowlight = uint8(clip(round(255.0 * (K_out ^ (1/2.2))), 0, 255))
       │
       ▼
 [Stage B: C4 Anti-Aliased Spatial Downsampling]
   1. Bicubic Resizing with Anti-Aliasing Filter (A7f):
      - Python: PIL.Image.BICUBIC or skimage.transform.resize(anti_aliasing=True) [Trap T5]
      - Downsamples to VGA (640x480) and QVGA (320x240)
       │
       ▼
 [Stage C: C5 Low-Bandwidth Streaming Modes]
   ├── Mode A (On-Device ROI Trace Streaming, B3):
   │   - Extracts spatial mean RGB traces for 3 ROIs on-device at 30 fps, 720p
   │   - Transmits 1D traces: 3 ROIs × 3 channels × 30 Hz × 16 bits = 4,320 bps = 4.32 kbps (over 115× below 500 kbps limit!)
   │
   └── Mode B (Degraded Video Transmission):
       - Frame-rate decimation: Subsamples to target fps in {10, 7.5, 6} fps.
         * Primary Protocol: Deterministic frame skipping [DOC §2.2]
         * Sensitivity Protocol: Boxcar temporal average-then-skip (A3)
       - Bitrate Compression: Re-encodes via FFmpeg libx264 CBR (-b:v 500k) / MATLAB VideoWriter MPEG-4 [VERIFY]
========================================================================================================================
```

#### Worked Example: Temporal Decimation Aliasing of Mains Flicker (A3)
Under artificial illumination powered by a $50\,\text{Hz}$ electrical grid, lights modulate at twice the line frequency ($100\,\text{Hz}$).
1. In a baseline video sampled at $f_s = 30\,\text{fps}$, the $100\,\text{Hz}$ flicker aliases to:

```math
100 \pmod{30} = 10.0 \text{ Hz}
```

   This $10.0\,\text{Hz}$ ripple lies far above the physiological cardiac band ($0.7\text{--}3.0\,\text{Hz}$, $42\text{--}180\,\text{bpm}$) and is cleanly removed by the 4th-order bandpass filter.
2. When the video is decimated via plain frame skipping to $f_s = 7.5\,\text{fps}$, the $10.0\,\text{Hz}$ component folds down:

```math
10.0 \pmod{7.5} = 2.5 \text{ Hz} \equiv 150 \text{ bpm}
```

   This falls directly inside the cardiac band, creating a catastrophic false cardiac peak!
3. When decimated to $f_s = 6.0\,\text{fps}$, the $10.0\,\text{Hz}$ component aliases to:

```math
|10.0 - 2 \times 6.0| = 2.0 \text{ Hz} \equiv 120 \text{ bpm}
```

   This also falls directly inside the cardiac band.
4. When decimated to $f_s = 10.0\,\text{fps}$:

```math
10.0 \pmod{10.0} = 0.0 \text{ Hz} \equiv \text{DC}
```

   The flicker folds to DC ($0\,\text{Hz}$), which is eliminated by mean centering.
*Mitigation:* Device-side boxcar temporal integration (average-then-skip) acts as a sinc prefilter, attenuating the $100\,\text{Hz}$ carrier before temporal decimation [TEAM-DEFINED].

#### Systematic Parameter Sweep Matrix

| Sweep Type | Dimension | Parameter Grid Evaluated | Baseline Reference |
| :--- | :--- | :--- | :--- |
| **OFAT 1 (C1: Lighting)** | Ambient Illuminance | Lux in {300, 150, 100, 50, 30, 20, 10} | 300 lux (Unconstrained) [DOC §2.3] |
| **OFAT 2 (C1: Gain/Noise)**| Auto-Exposure Mode | Dark-clip vs. Brightness-normalized-noisy | Dark-clip |
| **OFAT 3 (C4: Resolution)** | Spatial Dimension | 1280x720 (HD), 640x480 (VGA), 320x240 (QVGA) | 1280x720 (HD) [DOC §2.3] |
| **OFAT 4 (C5: Frame Rate)** | Temporal Sampling | f_s in {30, 15, 10, 7.5, 6} fps | 30 fps [DOC §2.3] |
| **OFAT 5 (C5: Bandwidth)** | Video Bitrate | Bitrate in {>= 5 Mbps, 2 Mbps, 1 Mbps, 500 kbps, 250 kbps} | >= 5 Mbps (Uncompressed) [DOC §2.3] |
| **Worst-Case Cell 1** | C1 + C4 | 50 lux + QVGA (320x240) @ 30 fps, 5 Mbps | Quantization + read noise |
| **Worst-Case Cell 2** | C1 + C5 | 50 lux + 720p @ 10 fps, 500 kbps | Low photon count + compression |
| **Worst-Case Cell 3** | C4 + C5 | 300 lux + QVGA (320x240) @ 10 fps, 500 kbps | Minimum spatial & temporal resolution |
| **Worst-Case Cell 4 (ALL)**| **C1+C2+C4+C5**| **50 lux + ITA-6 + QVGA + 10 fps + 500 kbps** | **The Ultimate Competition Stress Cell** |

---

## 4. Model and Training Plan

### 4.1 Incremental Baseline Ladder and Ablation Progression
To ensure falsifiability before PhD jurors [DOC §5.2], the system is evaluated across five progressive levels:
1. **Level 0: Naive Green-Channel rPPG:** Spatial mean of Green channel followed by bandpass filtering [DOC §2.1].
2. **Level 1: Classical State-of-the-Art (POS & CHROM):** Deterministic mathematical projections on spatial ROIs [DOC §2.1].
3. **Level 2: Public Deep rPPG Reference (PhysNet/TS-CAN):** Published deep architecture trained via standard ERM.
4. **Level 3: Multi-Task PhysNet-Lite with Waveform Decoder:** Separable 3D convs with dual outputs ($T$-length waveform + pooled embedding $\mathbf{z} \in \mathbb{R}^{128}$).
5. **Level 4: Full Proposed System (Tri-SQI Conditioned Stacking + Stratified Minimax):** The complete blueprint incorporating dynamic arbitration and fair minimax optimization across ITA bins [TEAM-DEFINED].

### 4.2 Multi-Objective Loss Formulations & Label Mappings (A9, A10)

```math
\mathcal{L}_{\text{total}} = \lambda_{\text{wave}} \mathcal{L}_{\text{wave}} + \lambda_{\text{spec}} \mathcal{L}_{\text{spec}} + \lambda_{\text{HR}} \mathcal{L}_{\text{HR-scalar}} + \lambda_{\text{SpO2}} \mathcal{L}_{\text{SpO2}} + \lambda_{\text{BP}} \mathcal{L}_{\text{BP}}
```

#### Loss Terms by Data Tier
* **Continuous Waveform Loss ($\mathcal{L}_{\text{wave}}$):** Evaluated strictly on Tier A (UBFC) and Tier B/C where continuous PPG waveforms exist:

```math
\mathcal{L}_{\text{wave}}(\mathbf{y}, \hat{\mathbf{y}}) = 1 - \frac{\sum_{t=1}^T (y_t - \mu_y)(\hat{y}_t - \mu_{\hat{y}})}{\sqrt{\sum_{t=1}^T (y_t - \mu_y)^2 \sum_{t=1}^T (\hat{y}_t - \mu_{\hat{y}})^2}}
```

* **Frequency Spectral Loss ($\mathcal{L}_{\text{spec}}$):** Cross-entropy between predicted PSD and ground-truth pulse spectrum.
* **Weakly-Supervised Scalar HR Loss ($\mathcal{L}_{\text{HR-scalar}}$, A9):** Used for Tier A `rppg_biases` where only scalar mean HR labels exist:

```math
\mathcal{L}_{\text{HR-scalar}} = \left| 60 \cdot \arg\max_{f \in [0.7, 3.0]} P_{\hat{y}}(f) - \text{HR}_{\text{label}} \right|
```

* **Heteroscedastic Uncertainty Loss for SpO2 and BP:**

```math
\mathcal{L}_{\text{param}}(y, \hat{y}, s) = \frac{1}{2} \exp(-s) |y - \hat{y}| + \frac{1}{2} s, \quad s = \log \hat{\sigma}^2
```

* **Group-DRO Minimax Fair Optimization (A10):**

```math
\min_{\theta} \max_{g \in \{1,\dots,6\}} \left\{ \mathcal{L}_g(\theta) + \rho \cdot \text{KL}(q_g \| p_g) \right\}
```

*Theoretical Reality Check (A10):* Mathematical convergence guarantees for Group-DRO hold strictly for convex losses. For deep neural networks, Group-DRO overfits small worst-case strata unless regularized or adjusted for sample count $n_g$. Furthermore, irreducible noise floors differ across skin tones due to photon absorption. We ablate standard ERM vs. Stratified Loss Reweighting ($w_g \propto 1/\sqrt{n_g}$) vs. Group-DRO [TEAM-DEFINED].

### 4.3 Calibration Strategies & Clinical Boundaries for SpO2 and BP (B6)
* **SpO2 Evaluation Protocol:**
  - Evaluated under both Calibration-Free (population constants) and Single-Point Calibrated regimes (10 s initial baseline offset).
  - Primary success criterion: Statistically beating the constant mean-predictor baseline with a subject-bootstrap confidence interval ($p < 0.05$).
  - Evaluates within-subject tracking metric ($\Delta \text{SpO}_2$ tracking accuracy).
  - Explicit reporting of reference-oximeter racial bias (citing Sjoding et al., *NEJM* 2020: clinical finger oximeters exhibit occult hypoxemia, overestimating saturation in dark skin).
* **Blood Pressure Evaluation Protocol:**
  - Evaluates three distinct regimes: (1) absolute SBP/DBP (mmHg), (2) change-from-calibration ($\Delta \text{BP}$), and (3) blood pressure category classification (Normotensive vs. Hypertensive).
  - Targets reported strictly relative to constant mean-predictor and demographics-only baselines ($\text{Age}, \text{BMI}$).
  - ISO 81060-2 cited strictly as an aspirational benchmark comparator [DOC §2.4, VERIFY].

### 4.4 Objective ITA Stratification & Dermatological Robustness (C2, A4)
* **Objective ITA Calculation (Trap T6, A4):** Converted from specular-free, shadow-free skin ROI pixels in CIE $1976\,L^*a^*b^*$:

```math
\text{ITA} = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}
```

* **Discretized ITA Strata (A4):** Bins are designated **ITA-1 through ITA-6** (never labeled as Fitzpatrick types):
  - **ITA-1 (Very Light):** $\text{ITA} > 55^\circ$
  - **ITA-2 (Light):** $41^\circ < \text{ITA} \le 55^\circ$
  - **ITA-3 (Intermediate):** $28^\circ < \text{ITA} \le 41^\circ$
  - **ITA-4 (Tan):** $10^\circ < \text{ITA} \le 28^\circ$
  - **ITA-5 (Brown):** $-30^\circ < \text{ITA} \le 10^\circ$
  - **ITA-6 (Dark):** $\text{ITA} \le -30^\circ$
* **Concordance & In-Frame Calibration:** Concordance among ITA, self/clinician Fitzpatrick, and Von Luschan tile reading is logged per subject. Official Figure 1 appears to show a color tile chart on the setup table [DOC §2.3 Fig 1]; we propose using it as an in-frame radiometric color reference [VERIFY against organizer protocol].
* **Vitiligo & Reflectance Anomalies:** Sub-ROIs undergo local Mahalanobis distance outlier detection in $a^*b^*$ color space. Sub-ROIs with bimodal distributions or extreme depigmentation are isolated, and the pipeline performs SNR-weighted spatial fusion. Synthetic depigmentation augmentation (Perlin-noise melanin attenuation) is applied during training.

---

## 5. Round-by-Round Plan and Submission Package

### 5.1 Round 1 Scope Reset & In-Depth Constraint Selection Matrix (B1)

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ROUND 1 CONSTRAINT SELECTION DECISION MATRIX (CORRECTED A8, B1)                                       │
├────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│ Constraint │ Data Avail. │ Sim. Rigor  │ Impl. Risk  │ Equity Align│ Jury Value  │ Selection Decision  │
│ Candidate  │ by 30 Nov   │ & Realism   │ & Feasibility│ with Theme  │ & Novelty   │                     │
├────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ C1: Low    │ High (Synth │ High (Linear│ Low (Clear  │ High (Rural │ Very High   │ SELECTED FOR DEEP   │
│ Light      │ + Pilot)    │ Poisson v2) │ physics)    │ healthcare) │ (Core rPPG) │ IMPLEMENTATION (R1) │
├────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ C4: Low    │ High (UBFC  │ High (Anti- │ Low (Resize │ Med (Budget │ High        │ SELECTED FOR DEEP   │
│ Resolution │ + synthetic)│ aliased AA) │ benchmarks) │ hardware)   │ (Graceful)  │ IMPLEMENTATION (R1) │
├────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ C2: Skin   │ Med/Low (A8:│ High (ITA   │ Med (Group- │ Maximum     │ Maximum     │ EVALUATION-ONLY ON  │
│ Tone / Vit │ rPPG-biases │ pipeline)   │ DRO tuning) │ (Core theme)│ (Fairness)  │ AVAILABLE STRATA    │
│            │ scalar only)│             │             │             │             │ (Full impl in R2)   │
├────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ C5: Low    │ High (Video │ High (CBR   │ Med (Mode A │ High (Remote│ High        │ SIMULATION SWEEP    │
│ Bandwidth  │ decimation) │ FFmpeg/WMV) │ trace mode) │ telehealth) │ (Spectral)  │ (Full impl in R3)   │
└────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘
```

*Round 1 Decision Rule [TEAM-DEFINED]:* Because `rppg_biases` provides only pre-cropped scalar HR data (A8), if the Tier B organizer clinical dataset agreement is not executed by **8 November 2026**, **C1 (Low Light)** and **C4 (Low Resolution)** take the two mandatory deep implementation slots for Round 1 [DOC §2.3 Task 1]. C2 is reported as an evaluation-only analysis across available public strata (reporting sample size $n$ per bin), and C5 is reported via full simulation sweeps (Mode A trace streaming and Mode B MPEG-4 compression).

#### IEEE-Format Report Outline (Mapped to Round 1 Criteria)
1. **Introduction & Innovation:** Theme alignment; hybrid classical-deep concept; falsifiable hypotheses.
2. **Methodology & Feasibility:** Classical CHROM/POS formulation; linear-light degradation simulator v2; anti-aliased spatial decimation; Mode A trace streaming.
3. **Experimental Setup & Thoroughness:** 5-fold subject-disjoint cross-validation on UBFC-rPPG; OFAT sweeps; real physical validation clips.
4. **Results & Error Discussion:** MAE, RMSE, Pearson $r$, repeated-measures Bland-Altman; C4 graceful degradation curves; C5 flicker-aliasing analysis.
5. **Documentation Clarity & Limitations:** Explicit research prototype boundaries; open items register.
*[ASSUMPTION: Report page limits assumed as 4 pages for Round 1, 6 pages for Final; to be verified].*

### 5.2 Phase-by-Phase Execution Plan (Reconciled A12, B1)

| Dimension | Round 1: Baseline HR + Initial Constraints [DOC §2.3] | Round 2: Two-Parameter (HR + SpO2) [DOC §2.3] | Round 3: Full System (HR + SpO2 + BP) [DOC §2.3] |
| :--- | :--- | :--- | :--- |
| **Submission Deadline** | **30 November 2026** (Results: 05 Dec) [DOC §6] | **16 January 2027** (Results: 21 Jan) [DOC §6] | **16–21 May 2027** (Live ICASSP Final) [DOC §6] |
| **Target Parameters** | Heart Rate (HR) in bpm | Joint HR (bpm) and SpO2 (%) | Simultaneous HR, SpO2, SBP/DBP (mmHg) |
| **Operating Conditions**| Controlled (>= 300 lux, 720p, 30 fps) & Uncontrolled | Full multi-constraint sweep (C1, C2, C4, C5) | Live edge streaming under >= 2 stress constraints |
| **Architecture Deltas** | Classical CHROM/POS + Tri-SQI + Simulators v2; Pretrained deep rPPG reference (B1) | Dual-head PhysNet-Lite + Stacked SpO2 Regressor + Stratified Minimax (ITA-1 to ITA-6) | Full multi-task morphology network + Demographics prior + ONNX/ARM64 edge engine |
| **Data & Experiments** | Tier A (UBFC) + Tier B/C initial clips; C1 & C4 deep; C5 sweep | Tier A + Tier B (Sri Ramachandra) + Tier C; SpO2 ablation (R/G vs R/B vs learned) | Complete clinical validation cohort + real physical camera stream at ICASSP |
| **Evaluation Metrics** | HR MAE, RMSE, r, Bland-Altman LoA, %CP5 | HR + SpO2 MAE, RMSE, LoA, %CP2, beating mean | Clinical error matrix; TOST equivalence test; Pi 4/5 FPS and latency profiling |
| **Deliverable Package** | Report (PDF), `Round1_Baseline_None.mat`, code | Extended report, `Round2_[Cond]_[Constr].mat`, code | Real-time deployable code, benchmark CSVs, live demo |
| **Exit Criteria** | HR MAE target <= 2.5 bpm / measured: [MEASURE] | SpO2 beating mean-predictor (p < 0.05); uncalib MAE target <= 3.5% / measured: [MEASURE] | BP beating demographics-only baseline; Pi 4 FPS target >= 20 / measured: [MEASURE] |

### 5.3 Required Submission Files and Data Structure Compliance
All deliverables adhere strictly to official competition data organization [DOC §1.2, §3]:

```text
FaceVitals_Submission_Package/
├── /Round1_Baseline/                           [DOC §1.2]
│   ├── Round1_Baseline_None.mat                [DOC §1.2] (Facial video & ref HR at baseline)
│   ├── Round1_LowLight_C1.mat                  [ASSUMPTION: Proposed for uncontrolled condition requirement]
│   ├── Round1_LowResolution_C4.mat             [ASSUMPTION: Proposed for uncontrolled condition requirement]
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
```

#### Mandatory Internal Variables in Every `.mat` File [DOC §1.2]
1. `facial_video`: Raw RGB video frames of the subject $[T \times H \times W \times 3]$ (`uint8`, stored in HDF5 `-v7.3` format to avoid 2GB corruption, Trap T11).
2. `roi_data`: Extracted facial ROI coordinates/masks (forehead, cheeks) $[T \times 3 \times 4]$.
3. `reference_signal`: Ground-truth physiological signal(s) aligned at video frame rate.
4. `processed_signal`: Output of the participant's extraction algorithm.
5. `metrics`: Struct containing scalar evaluation results: `metrics.MAE`, `metrics.RMSE`, `metrics.Pearson_r`, `metrics.BlandAltman_LoA` $[2 \times 1]$.
6. `params`: Struct containing operational metadata:
   - `params.camera_resolution`: e.g., `'1280x720'` or `'320x240'`
   - `params.frame_rate`: e.g., `30` or `10`
   - `params.illuminance_lux`: e.g., `50` or `300`
   - `params.camera_to_subject_distance`: e.g., `'50cm'`
   - `params.subject_fitzpatrick_skin_type`: e.g., `'TypeV'` [DOC §1.2 field name preserved]
   - `params.subject_age`: e.g., `64`
   - `params.constraint_applied`: e.g., `'C1, C4'`
   - `params.extra`: Struct containing team-defined tracking fields [DOC §1.2, TEAM-DEFINED]: `params.extra.achieved_bitrate_kbps`, `params.extra.measured_ita_deg`, `params.extra.ita_stratum` (`'ITA-1'` to `'ITA-6'`), `params.extra.c5_mode` (`'ModeA'` or `'ModeB'`), `params.extra.coverage_fraction`.

### 5.4 Submission Verification Checklist (Pre-Flight Gate)
```markdown
[ ] 1. Identifier Consistency: Every subject ID in target_video.mp4 matches reference_signal.csv, 
       processed_output.csv, and RoundX_Condition_Constraint.mat [DOC §1.2].
[ ] 2. Variable Exactness: Every .mat file contains all 6 required variables with identical spelling [DOC §1.2].
[ ] 3. Data Type & File Size: facial_video stored as uint8 in HDF5 (-v7.3) format to prevent 2GB corruption [Trap T11].
[ ] 4. Metric Reproducibility: Executing process_roundX.m reproduces metrics struct exactly from raw video.
[ ] 5. Directory Compliance: Root directories adhere to /Round1_Baseline/, /Round2_HR_SpO2/, etc. [DOC §1.2].
[ ] 6. Ethics Documentation: Sri Ramachandra legal agreement or institutional IRB clearance attached [DOC §3].
[ ] 7. Roster Verification: 1 Faculty, 3–10 Undergrads (>= 3 SPS student members verified active) [DOC §5.1].
```

---

## 6. Evaluation Protocol

### 6.1 Clinical Metric Formulations and Statistical Rigor
Evaluation adheres strictly to biomedical method-comparison standards (Altman & Bland, 1983 [DOC §10]):
1. **Mean Absolute Error (MAE):**

```math
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
```

2. **Root Mean Square Error (RMSE):**

```math
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}
```

3. **Pearson Correlation Coefficient ($r$):**

```math
r = \frac{\sum_{i=1}^N (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^N (y_i - \bar{y})^2 \sum_{i=1}^N (\hat{y}_i - \bar{\hat{y}})^2}}
```

4. **Repeated-Measures Bland–Altman Agreement Limits (Trap T8):**
   * *Mean Bias:* $\bar{d} = \frac{1}{N}\sum (y_i - \hat{y}_i)$
   * *Variance Decomposition:* To account for multiple sliding windows per subject, variance is partitioned into between-subject and within-subject components: $s_d^2 = \sigma_{\text{between}}^2 + \sigma_{\text{within}}^2$.
   * *Limits of Agreement (LoA):* $\bar{d} \pm 1.96 \cdot s_d$, with 95% confidence intervals derived via 2,000 subject-level cluster bootstrap iterations.
5. **Percentage Within Clinical Tolerance:**
   * $\text{HR} \le \pm 5\,\text{bpm}$ (`%CP5`): Percentage of estimates within 5 bpm of reference.
   * $\text{SpO}_2 \le \pm 2\%$ (`%CP2`) and $\le \pm 3\%$ (`%CP3`).

### 6.2 The C2 Equivalence Testing & Fairness Specification (B7)
The official document mandates: *"Per-type error must not deviate beyond a pre-specified tolerance from the population mean across all six types"* [DOC §2.2].

#### Mathematical Equivalence Formulation (TOST, B7)
Rather than a naive point-estimate check, the fairness test is formulated as a Two-One-Sided Equivalence Test (TOST). The criterion passes if and only if the **upper bound of the 95% subject-bootstrap confidence interval** of the stratum deviation lies within tolerance $\tau$:

```math
\text{Upper}_{95\%}\left( \left| \text{MAE}_g - \mu_{\text{pop}} \right| \right) \le \tau
```

Both unweighted stratum mean and subject-weighted cohort mean are reported:

```math
\mu_{\text{unweighted}} = \frac{1}{6} \sum_{g=1}^6 \text{MAE}_g, \quad \mu_{\text{weighted}} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
```

Both absolute tolerance ($\tau_{\text{abs}} = 1.5\,\text{bpm}$ [PROVISIONAL]) and relative tolerance ($\tau_{\text{rel}} = \tau / \mu_{\text{pop}} = 20\%$ [PROVISIONAL]) are evaluated.

#### Mondrian Conformal Prediction Intervals (B7)
To ensure that predicted 95% confidence bounds are valid across all skin tones, we compute Mondrian conformal prediction intervals calibrated independently within each ITA stratum:

```math
\hat{C}_g(X) = \left[ \hat{y} - q_{1-\alpha, g}, \, \hat{y} + q_{1-\alpha, g} \right]
```

where $q_{1-\alpha, g}$ is the empirical $(1-\alpha)$ quantile of non-conformity scores within stratum $g$.

### 6.3 Performance Target Matrix (Baseline-Relative & Measured Targets, B6)

| Evaluation Dimension | Parameter & Metric | Baseline Reference Target | Stress Target (<= 50 lux, QVGA, <= 10 fps) | Fallback Action if Missed [TEAM-DEFINED] |
| :--- | :--- | :--- | :--- | :--- |
| **1. Heart Rate (HR)** | MAE (bpm) | Beating naive mean; target <= 1.5 bpm / measured: [MEASURE] | target <= 3.5 bpm / measured: [MEASURE] | Expand window to 15 s; fallback to POS |
| [DOC §2.3 Task 1] | Pearson r | target >= 0.95 / measured: [MEASURE] | target >= 0.85 / measured: [MEASURE] | Revert to classical POS branch |
| | % within +-5 bpm | target >= 95.0% / measured: [MEASURE] | target >= 82.0% / measured: [MEASURE] | Increase SQI abstention threshold |
| **2. SpO2 Saturation** | MAE (%) | Beating naive mean (p < 0.05); target <= 1.8% / measured: [MEASURE] | target <= 3.2% / measured: [MEASURE] | Require single-point calibration |
| [DOC §2.3 Round 2] | Calibration-Free | target <= 3.0% / measured: [MEASURE] | target <= 4.5% / measured: [MEASURE] | Abstain if SQI_abstain < 0.60 |
| | % within +-2% | target >= 85.0% / measured: [MEASURE] | target >= 70.0% / measured: [MEASURE] | Restrict to calibrated mode |
| **3. Blood Pressure** | SBP MAE (mmHg) | Beating demo prior; target <= 5.0 mmHg / measured: [MEASURE] | target <= 8.0 mmHg / measured: [MEASURE] | Report category (Normotensive) |
| [DOC §2.3 Round 3] | DBP MAE (mmHg) | Beating demo prior; target <= 4.0 mmHg / measured: [MEASURE] | target <= 6.5 mmHg / measured: [MEASURE] | Label "non-video-derived baseline" |
| | ISO 81060-2 Match | Aspirational comparator only [DOC §2.4, VERIFY] | Aspirational comparator only [VERIFY] | Explicit research limitation statement |
| **4. C2 Skin Fairness** | Equivalence Bound | Upper 95% CI |Delta| <= 1.5 bpm [PROVISIONAL] | Upper 95% CI |Delta| <= 2.0 bpm [PROVISIONAL] | Adjust group reweighting w_g |
| [DOC §2.2] | Min Stratum n | n_g >= 15 subjects | n_g >= 15 subjects | Flag "UNDERPOWERED" (no pass/fail) |
| **5. C5 Streaming** | Mode A Bandwidth | 4,320 bps (4.32 kbps) | 4,320 bps (4.32 kbps, >115× below 500 kbps limit) | Preferred remote operational mode |
| [DOC §2.2 C5, B3] | Mode B Bandwidth | Achieved <= 500 kbps (MPEG-4 video stream) | Achieved <= 500 kbps | Frame skipping + CBR rate control |
| **6. Gating & Failure** | Abstention Coverage | 100% (No windows discarded) | target >= 75.0% accepted / measured: [MEASURE] | Plot risk-coverage curve (Trap T7) |
| [Trap T7] | Risk-Coverage AUC | target <= 1.2 bpm / measured: [MEASURE] | target <= 2.8 bpm / measured: [MEASURE] | Concurrently report ungated MAE |
| **7. Edge Efficiency** | Raspberry Pi 4/5 Latency| Pi 5: target <= 25 ms; Pi 4: target <= 40 ms / measured: [MEASURE] | Pi 4: target <= 45 ms / measured: [MEASURE] | Stride deep trunk to every 30 frames |
| [DOC §2.1, §5.2] | Edge Throughput | Pi 5: target >= 35 fps; Pi 4: target >= 22 fps / measured: [MEASURE] | Pi 4: target >= 20 fps / measured: [MEASURE] | Switch to pure classical POS fallback |

---

## 7. Comprehensive Risk Register

| Risk Description | P | S | Early-Warning Indicator | Architectural Mitigation | Fallback Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1. Low-Light SNR Collapse (<= 50 lux)** | H | H | Pixel variance drops below read noise floor sigma_read^2. | Bilateral spatial filtering prior to projection; no independent per-channel CLAHE. | Auto-expand window to 15 s; fallback to Green-difference rPPG. |
| **R2. Skin-Tone Bias Drift on Darker Bins** | H | H | TOST upper bound > 1.5 bpm on validation folds. | Stratified loss reweighting / Group-DRO; sub-ROI SNR weighting on forehead. | Exclude depigmented patches; state sample size n_g transparently. |
| **R3. Low-Res Landmark Jitter / Failure** | M | H | Landmark Euclidean jitter > 5 px across frames at QVGA. | Kalman-filtered bounding box; multi-scale landmark initialization; 12x12 px floor. | Revert to full-face bounding box spatial averaging if conf < 0.4. |
| **R4. 10 fps Decimation & Mains Aliasing**| H | H | Spurious PSD peaks at flicker beat frequencies (2.5 Hz / 2.0 Hz). | Boxcar temporal averaging (average-then-skip sinc prefilter); Lomb-Scargle; harmonic check. | Flag flicker harmonic; report robust 10-second windowed median. |
| **R5. Uncalibrated SpO2/BP Drift** | H | H | Test error fails to beat constant mean-predictor baseline. | Dual-mode reporting (calibrated vs. uncalibrated); stacked physics-latent regressor. | Explicitly report physics caveats; label demographic output. |
| **R6. Absence of Public Video BP Labels** | H | H | No continuous arterial BP labels in public rPPG datasets. | Early execution of Sri Ramachandra agreement; pretrain morphology on PulseDB. | Benchmark BP on Tier B clinical data; use pulse morphology ML head. |
| **R7. Tier B Legal Access Delays** | M | H | No response on Sri Ramachandra agreement by 8 Nov 2026. | Multi-tier fallback: swap C4 into deep slot for Round 1; C2 evaluation-only (B1). | Rely on Tier A (UBFC) + local Tier C IRB data. |
| **R8. Simulation-to-Reality Gap** | M | M | Synthetic-trained model collapses on real 50-lux validation clips. | Linear-light simulator v2 with auto-exposure normalization; hold out unseen lux settings. | Calibrate photon parameters against physical lux meter data. |
| **R9. Cherry-Picking via Gating (Trap T7)**| M | M | Low reported MAE but coverage drops below 50%. | Mandatory concurrent reporting of ungated and gated results; plot risk-coverage curves. | Plot full risk-coverage curves across coverage fractions 0.1 to 1.0. |
| **R10. Raspberry Pi 3D-Conv Latency** | M | M | Pi 4 inference exceeds 45 ms/frame (< 20 fps). | INT8 QAT compilation via ONNX Runtime / NEON; stride deep trunk to every 30 frames. | Fallback to 2D+temporal shift (TS-CAN) or pure classical POS. |
| **R11. Reference-Oximeter Bias (B6)** | M | M | Reference pulse oximeter overestimates SpO2 on ITA-5/6. | Cross-reference against arterial blood gas if available; document Sjoding bias. | Disclose oximeter reference inaccuracy in clinical limitation section. |
| **R12. Window-Length Confound (A1)** | M | M | Inconsistent frequency resolution across decimated frame rates. | Enforce identical 10.0 s wall-clock window across all fps; isolate adaptive windowing. | Explicitly label window duration and Delta-f for every result. |
| **R13. File-Format Non-Compliance** | M | H | Submission validator rejects folder naming or missing .mat vars. | Automated CI validator (`verify_submission.py`) run prior to upload. | Lock submission package 48 hours prior to deadline. |
| **R14. Team Roster Lock After 25 Oct** | L | H | Student membership lapses or member change attempted. | Faculty verifies IEEE SPS student membership numbers on IEEE portal by 22 Oct. | Freeze roster of 5 committed undergrads + 1 faculty + 1 tutor by 22 Oct. |

---

## 8. Timeline and Execution Strategy

### 8.1 Critical Official Milestones Timeline
* **17 October 2026:** Planning Date & Architecture Kickoff.
* **25 October 2026:** Official Team Registration Deadline [DOC §6] *(8 days remaining!)*.
* **30 November 2026:** Round 1 Submission Deadline [DOC §6].
* **05 December 2026:** Round 1 Result Declaration [DOC §6].
* **16 January 2027:** Round 2 Submission Deadline [DOC §6].
* **21 January 2027:** Round 2 Result Declaration [DOC §6].
* **21 Jan – 01 Feb 2027:** Finalist Selection Window [DOC §6].
* **09 February 2027:** Announcement of 3 Finalist Teams [DOC §6].
* **16–21 May 2027:** Live Final Competition at IEEE ICASSP 2027, New Orleans, USA [DOC §6].

### 8.2 Immediate First-8-Days Action Checklist (Strictly 17 Oct – 24 Oct 2026, A12)

```markdown
- [ ] Day 1 (17 Oct): Convene team. Audit roster: verify 1 Faculty Supervisor, >= 3 undergraduate 
                      IEEE SPS student memberships active on IEEE portal [DOC §5.1].
- [ ] Day 2 (18 Oct): Draft and email official legal agreement request for Sri Ramachandra dataset 
                      to Dr. Vijay Jeyakumar (vijayjeyakumar@ieee.org), copying 
                      sp-competitions@listserv.ieee.org [DOC §3, §8]. Initiate external BP requests (BP4D+/V4V).
- [ ] Day 3 (19 Oct): All team members execute 'SP and VIP Cup Terms & Conditions' and 
                      IEEE Conflict-of-Interest forms [DOC §5.3, §7.1].
- [ ] Day 4 (20 Oct): Apply for complimentary MathWorks Student Competition MATLAB Licenses 
                      via official Dropbox repository [DOC §7.2].
- [ ] Day 5 (21 Oct): Download Tier A public datasets: UBFC-rPPG (Datasets 1 & 2) and 
                      AiPEX-Lab/rppg_biases. Verify file checksums and local disk layout.
- [ ] Day 6 (22 Oct): Submit institutional IRB application for Tier C supplementary vital-sign data 
                      collection protocol adapting Sri Ramachandra clinical guidelines [DOC §3].
- [ ] Day 7 (23 Oct): Initialize Git repository with automated formatting, CI testing, and 
                      pre-submission verification scripts. Launch Week 1 edge profiling spike on Pi 4/5.
- [ ] Day 8 (24 Oct): Complete official team registration on portal:
                      https://www2.securecms.com/SPCup/SPCRegistration.asp [DOC §7.1].
                      CONFIRM REGISTRATION RECEIPT PRIOR TO MIDNIGHT 25 OCT DEADLINE.
```

### 8.3 Team Roles & Work Breakdown Structure (1 Faculty, 1 Tutor, 5 Undergrads)
* **Faculty Supervisor (PI):** Institutional compliance, legal agreement execution, IRB oversight, official correspondence with IEEE SPS [DOC §5.1].
* **Graduate Student (Tutor):** Co-supervises architecture, reviews statistical verification, oversees PyTorch-to-ONNX compilation [DOC §5.1].
* **Undergrad 1 (Edge & Embedded Architect):** Edge profiling spike (B8), Raspberry Pi 4/5 runtime, ONNX NEON optimization, C5 Mode A trace streaming engine.
* **Undergrad 2 (Computer Vision & Preprocessing):** Landmark tracking, multi-ROI extraction, C4 anti-aliased resize pipeline, C7 occlusion fallback.
* **Undergrad 3 (Classical Signal Processing):** POS/CHROM pipelines, causal IIR filter bank, Welch/Lomb-Scargle PSD, Tri-SQI engine.
* **Undergrad 4 (Deep Learning & Fairness):** PhysNet-Lite training, multi-task heads, stratified loss reweighting / Group-DRO, vitiligo synthesis.
* **Undergrad 5 (Data Engineering & Evaluation):** Simulator v2 (linear light, digital gain, order ablation), ground-truth alignment, Bland-Altman statistics, `.mat` package validator.

### 8.4 Weekly Work Packages & Effort Allocation across Constraints (B9)

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ WEEKLY WORK PACKAGES: 17 OCT 2026 TO 21 MAY 2027                                                      │
├──────────────┬──────────────────────────────────────────────────────────────────┬──────────────────────┤
│ Time Window  │ Core Technical Work Packages                                     │ Milestone / Freeze   │
├──────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Weeks 1–2    │ Edge profiling spike (Pi 4/5); Tier A dataset ingestion;        │ 24 Oct: Roster freeze│
│ (17–31 Oct)  │ Simulator v2 implementation; Classical POS/CHROM baseline.       │ 31 Oct: Edge spike go│
├──────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Weeks 3–4    │ Full C1 & C4 OFAT sweeps; Mode A trace streaming; Tri-SQI engine;│ 08 Nov: Tier B gate  │
│ (01–14 Nov)  │ Pretrained deep rPPG reference integration; IRB Tier C launch.   │ 14 Nov: Code freeze  │
├──────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Weeks 5–6    │ Round 1 report drafting; .mat verification; cluster bootstrap;   │ 24 Nov: Doc freeze   │
│ (15–30 Nov)  │ Full Round 1 package submission on official portal.              │ 30 Nov: R1 SUBMISSION│
├──────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Weeks 7–9    │ PhysNet-Lite multi-task training; SpO2 ratio-of-ratios ablation; │ 05 Dec: R1 results   │
│ (01–21 Dec)  │ Group-DRO tuning over ITA-1 to ITA-6; Tier B data ingestion.     │ 21 Dec: Arch freeze  │
├──────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Weeks 10–12  │ Joint HR+SpO2 multi-constraint stress testing; Conformal bounds; │ 10 Jan: R2 freeze    │
│ (22 Dec–16Jan│ Round 2 technical report & multi-condition package upload.       │ 16 Jan: R2 SUBMISSION│
├──────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Weeks 13–15  │ BP head morphology integration; PulseDB pretraining;             │ 21 Jan: R2 results   │
│ (17 Jan–09Feb│ Evaluation against demographics-only baseline; Finalist selec.   │ 09 Feb: Finalist Ann.│
├──────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Weeks 16–22  │ Embedded C++ deployment on Raspberry Pi 5; Live touchscreen GUI; │ 01 May: Hardware lock│
│ (10 Feb–01May│ Clinical validation trials on Tier B/C cohorts; Presentation.    │ 10 May: Travel prep  │
├──────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────┤
│ Final Week   │ IEEE ICASSP 2027 Final Competition (New Orleans, USA):           │ 16–21 May 2027:      │
│ (16–21 May)  │ In-person presentation and live hardware demonstration.          │ LIVE ICASSP FINAL    │
└──────────────┴──────────────────────────────────────────────────────────────────┴──────────────────────┘
```

#### Percentage Effort Allocation across Constraints
* **Round 1 (Baseline & Initial Constraints):**
  - **C1 (Low Light):** 35% (Linear-light simulator v2, auto-exposure normalization, noise floor calibration).
  - **C4 (Low Resolution):** 30% (Anti-aliasing downsampling parity, minimum-ROI floor, graceful degradation curves).
  - **C2 (Skin Tone):** 15% (ITA calculation pipeline, initial concordance analysis on public strata).
  - **C5 (Low Bandwidth):** 20% (Mode A trace streaming architecture, frame decimation aliasing analysis).
  *Rationale:* C1 and C4 are completely simulation-verifiable on public video data by 30 Nov 2026.
* **Round 2 (Joint HR + SpO2):**
  - **C1:** 15% | **C4:** 15% | **C2:** 45% (Full Stratified Minimax / Group-DRO across ITA-1 to ITA-6, vitiligo detection) | **C5:** 25% (MPEG-4 rate control, SpO2 chroma-subsampling degradation).
* **Round 3 (Full Clinical System & Edge Demo):**
  - **C1:** 15% | **C2:** 25% | **C4:** 15% | **C5:** 45% (Embedded Raspberry Pi 4/5 deployment, real-time live camera streaming under stress).

---

## 9. Code and Reproducibility

### 9.1 Repository Structure

```text
FaceVitals_Core/
├── config/
│   ├── baseline_config.yaml
│   ├── eval_protocol.yaml                    # Frozen pre-registered protocol
│   └── stress_c1_c2_c4_c5.yaml               # $C_1, C_2, C_4, C_5$ sweeps
├── src/
│   ├── acquisition/
│   │   ├── stream_reader.py
│   │   └── metadata_logger.py
│   ├── preprocessing/
│   │   ├── face_roi_tracker.py
│   │   ├── ita_classifier.py                # $\text{ITA}$ stratification (ITA-1 to ITA-6)
│   │   └── degradation_sim.py               # $C_1, C_4, C_5$ physics simulator v2
│   ├── classical/
│   │   ├── rppg_pos.py                      # $\mathbf{S}_{\text{POS}}$ projection
│   │   ├── rppg_chrom.py                    # $\text{CHROM}$ projection
│   │   └── filter_bank.py                   # Causal IIR $[0.7, 3.0]\,\text{Hz}$
│   ├── models/
│   │   ├── physnet_lite.py                  # Spatiotemporal trunk ($\approx 0.85\,\text{M}$ params)
│   │   ├── multi_task_heads.py              # $\text{HR}$, $\text{SpO}_2$, $\text{BP}$ heads
│   │   └── group_dro_loss.py                # Minimax fairness ($q_g$)
│   ├── fusion/
│   │   ├── sqi_engine.py                    # Tri-SQI (${\text{SQI}}_{\text{fuse}}, {\text{SQI}}_{\text{win}}, {\text{SQI}}_{\text{abstain}}$)
│   │   └── dynamic_arbiter.py               # Learned stacking gate $\alpha({\text{SQI}})$
│   └── evaluation/
│       ├── metrics_bland_altman.py          # $\text{MAE}$, $\text{RMSE}$, $r$, repeated-measures LoA
│       ├── equivalence_tost.py              # TOST fairness equivalence testing
│       ├── bootstrap_ci.py                  # Subject-level cluster bootstrap
│       └── mat_export_validator.py          # Compliant HDF5 `.mat` export
├── scripts/
│   ├── run_round1_experiments.py
│   ├── edge_profile_spike.py
│   └── package_submission.py
└── tests/
    └── test_mat_format.py
```

### 9.2 Complete Algorithmic Implementations (B9)

#### 1. Multi-ROI Extraction with Landmark Gating & Fallback
```python
import cv2
import numpy as np
from typing import Dict, Tuple, Any

def extract_facial_rois(frame_rgb: np.ndarray, 
                       face_mesh_detector: Any, 
                       cascade_fallback: cv2.CascadeClassifier,
                       target_res_scale: float = 1.0) -> Dict[str, Any]:
    """
    Extracts Forehead (60x30), Left Cheek (25x25), and Right Cheek (25x25) ROIs [DOC §10].
    Gracefully degrades under low resolution (C4) down to a 12x12 px floor.
    """
    h, w = frame_rgb.shape[:2]
    fh_w, fh_h = max(12, int(60 * target_res_scale)), max(12, int(30 * target_res_scale))
    ck_w, ck_h = max(12, int(25 * target_res_scale)), max(12, int(25 * target_res_scale))
    
    # 1. Attempt facial landmark mesh
    results = face_mesh_detector.process(frame_rgb)
    if results and results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        fh_c = (int(lm[10].x * w), int(lm[10].y * h) + int(0.03 * h))
        lc_c = (int(lm[116].x * w), int(lm[116].y * h))
        rc_c = (int(lm[345].x * w), int(lm[345].y * h))
        conf = 0.95
    else:
        # 2. Fallback to Haar Cascade
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = cascade_fallback.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            fh_c = (fx + int(fw * 0.50), fy + int(fh * 0.22))
            lc_c = (fx + int(fw * 0.28), fy + int(fh * 0.65))
            rc_c = (fx + int(fw * 0.72), fy + int(fh * 0.65))
            conf = 0.65
        else:
            # 3. Central heuristic fallback
            fh_c, lc_c, rc_c = (int(w * 0.5), int(h * 0.25)), (int(w * 0.3), int(h * 0.55)), (int(w * 0.7), int(h * 0.55))
            conf = 0.20

    def crop_box(cx, cy, bw, bh):
        x1 = max(0, min(w - bw, cx - bw // 2))
        y1 = max(0, min(h - bh, cy - bh // 2))
        return frame_rgb[y1:y1+bh, x1:x1+bw], [int(x1), int(y1), int(bw), int(bh)]

    fh_crop, fh_box = crop_box(fh_c[0], fh_c[1], fh_w, fh_h)
    lc_crop, lc_box = crop_box(lc_c[0], lc_c[1], ck_w, ck_h)
    rc_crop, rc_box = crop_box(rc_c[0], rc_c[1], ck_w, ck_h)

    return {
        "forehead": fh_crop, "left_cheek": lc_crop, "right_cheek": rc_crop,
        "bboxes": {"forehead": fh_box, "left_cheek": lc_box, "right_cheek": rc_box},
        "landmark_confidence": conf, "tracking_success": conf >= 0.40
    }
```

#### 2. Physics-Based Linear-Light Degradation Simulator v2 (C1 Engine, A7, B5)
```python
def simulate_low_light_v2(frame_srgb: np.ndarray, target_lux: float,
                          base_lux: float = 300.0,
                          phi_full_well: float = 5000.0,
                          read_noise_std: float = 2.5,
                          auto_exposure_mode: str = "normalized_noisy",
                          seed: int = 42) -> np.ndarray:
    """
    Simulates physical photon shot noise and read noise in linear irradiance space (A7).
    Modes:
      - 'dark_clip': Raw attenuated low-light frame.
      - 'normalized_noisy': Auto-exposure boosted frame with amplified noise floor.
    """
    assert frame_srgb.dtype == np.uint8
    np.random.seed(seed)
    
    # 1. Linearize sRGB to relative physical radiance
    frame_lin = np.power(frame_srgb.astype(np.float32) / 255.0, 2.2)
    
    # 2. Multiplicative illuminance attenuation
    gain = np.clip(target_lux / base_lux, 0.0, 1.0)
    atten_lin = frame_lin * gain
    
    # 3. Photon conversion & Poisson shot noise draw (Trap T1)
    mean_photons = atten_lin * phi_full_well
    shot_noise = np.random.poisson(mean_photons).astype(np.float32)
    
    # 4. Sensor Gaussian read noise
    read_noise = np.random.normal(0.0, read_noise_std, size=frame_srgb.shape).astype(np.float32)
    total_photons = shot_noise + read_noise
    
    # 5. Auto-exposure digital gain emulation (A7b)
    if auto_exposure_mode == "normalized_noisy":
        digital_gain = 1.0 / max(1e-4, gain)
        norm_lin = np.clip((total_photons / phi_full_well) * digital_gain, 0.0, 1.0)
    else:
        norm_lin = np.clip(total_photons / phi_full_well, 0.0, 1.0)
        
    # 6. Gamma re-encoding & 8-bit quantization (A7a)
    out_srgb = np.clip(np.round(255.0 * np.power(norm_lin, 1.0 / 2.2)), 0, 255)
    return out_srgb.astype(np.uint8)
```

#### 3. Objective ITA Calculation and Skin-Tone Stratification (C2 Engine, A4)
```python
def compute_ita_stratum(roi_rgb: np.ndarray) -> Tuple[float, str]:
    """
    Computes Individual Typology Angle (ITA) in CIE 1976 L*a*b* space (Trap T6, A4).
    Returns median ITA and designated stratum 'ITA-1' through 'ITA-6'.
    """
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0
    
    # Mask out extreme specular highlights (L > 95) and deep cast shadows (L < 15)
    valid = (L >= 15.0) & (L <= 95.0)
    if np.sum(valid) < 25:
        return 0.0, "ITA-Undefined"
        
    med_L = float(np.median(L[valid]))
    med_b = float(np.median(b[valid]))
    denom = med_b if abs(med_b) > 1e-4 else 1e-4
    ita_deg = float(np.arctan((med_L - 50.0) / denom) * (180.0 / np.pi))
    
    if ita_deg > 55.0: stratum = "ITA-1"
    elif ita_deg > 41.0: stratum = "ITA-2"
    elif ita_deg > 28.0: stratum = "ITA-3"
    elif ita_deg > 10.0: stratum = "ITA-4"
    elif ita_deg > -30.0: stratum = "ITA-5"
    else: stratum = "ITA-6"
        
    return ita_deg, stratum
```

#### 4. Anti-Aliased Resizing & C5 Decimation Modes (A3, A7f, B3)
```python
from PIL import Image

def resize_anti_aliased(frame_rgb: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    """Bicubic downsampling with anti-aliasing prefilter matching MATLAB imresize (Trap T5)."""
    pil_img = Image.fromarray(frame_rgb)
    resized_pil = pil_img.resize(target_wh, resample=Image.Resampling.BICUBIC)
    return np.array(resized_pil)

def decimate_temporal_c5(frames: list, orig_fps: float, target_fps: float, 
                         mode: str = "frame_skip") -> Tuple[list, float]:
    """
    Temporal decimation for C5 (A3):
      - 'frame_skip': Official competition protocol [DOC §2.2]
      - 'average_then_skip': Sinc prefilter mitigating mains flicker aliasing
    """
    stride = orig_fps / target_fps
    num_out = int(len(frames) / stride)
    indices = [int(round(i * stride)) for i in range(num_out) if int(round(i * stride)) < len(frames)]
    
    if mode == "frame_skip":
        return [frames[idx] for idx in indices], target_fps
    elif mode == "average_then_skip":
        out = []
        for i in range(len(indices)):
            s_idx = indices[i]
            e_idx = indices[i+1] if i+1 < len(indices) else min(len(frames), s_idx + int(stride))
            out.append(np.mean(frames[s_idx:max(s_idx+1, e_idx)], axis=0).astype(np.uint8))
        return out, target_fps
    raise ValueError(f"Unknown mode {mode}")
```

#### 5. Real-Time Causal POS rPPG Waveform Extraction & Peak Tracking with Filter Bank
```python
from scipy.signal import butter, sosfilt

class CausalPOSStreamer:
    """Real-time causal POS rPPG extractor for edge streaming."""
    def __init__(self, fps: float = 30.0, win_sec: float = 1.6):
        self.fps = fps
        self.win_len = max(8, int(fps * win_sec))
        self.buffer = []
        nyq = fps / 2.0
        high = min(3.0, nyq - 0.1)
        self.sos = butter(4, [0.7, high], btype='bandpass', fs=fps, output='sos')
        self.zi = np.zeros((self.sos.shape[0], 2))
        
    def step(self, roi_mean_rgb: np.ndarray) -> Tuple[float, float]:
        self.buffer.append(roi_mean_rgb.astype(np.float64))
        if len(self.buffer) > self.win_len:
            self.buffer.pop(0)
        if len(self.buffer) < self.win_len:
            return 0.0, 0.0
            
        C = np.array(self.buffer).T # [3 x W]
        Cn = C / (np.mean(C, axis=1, keepdims=True) + 1e-6) - 1.0
        S1 = Cn[0, :] - Cn[1, :]
        S2 = Cn[0, :] + Cn[1, :] - 2.0 * Cn[2, :]
        alpha = np.std(S1) / (np.std(S2) + 1e-6)
        h = S1[-1] + alpha * S2[-1]
        
        filtered, self.zi = sosfilt(self.sos, [h], zi=self.zi)
        power = float(np.var(S1 + alpha * S2))
        return float(filtered[0]), power
```

#### 6. Multi-Task Forward Pass (PhysNet-Lite + Heads, B4)
```python
import torch
import torch.nn as nn

class MultiTaskPhysNet(nn.Module):
    """Deep spatiotemporal rPPG architecture with waveform decoder and multi-task heads."""
    def __init__(self, trunk_model: nn.Module, latent_dim: int = 128):
        super().__init__()
        self.trunk = trunk_model # Outputs waveform (B, T) and latent z (B, 128)
        self.hr_head = nn.Linear(latent_dim, 2) # [mean, log_var]
        self.spo2_head = nn.Linear(latent_dim + 1, 2) # [mean, log_var] (concatenates R_phys)
        self.bp_head = nn.Linear(latent_dim + 2, 4) # [sbp_mean, sbp_logvar, dbp_mean, dbp_logvar] (concatenates Age, BMI)

    def forward(self, x: torch.Tensor, r_phys: torch.Tensor, demo_non_ita: torch.Tensor) -> dict:
        waveform, z = self.trunk(x)
        hr_out = self.hr_head(z)
        hr_pred = hr_out[:, 0] * 140.0 + 40.0
        
        spo2_in = torch.cat([z, r_phys], dim=-1)
        spo2_out = self.spo2_head(spo2_in)
        spo2_pred = torch.sigmoid(spo2_out[:, 0]) * 30.0 + 70.0
        
        bp_in = torch.cat([z, demo_non_ita], dim=-1) # STRICTLY EXCLUDES ITA (A6)
        bp_out = self.bp_head(bp_in)
        sbp_pred = bp_out[:, 0] * 80.0 + 80.0
        dbp_pred = bp_out[:, 2] * 50.0 + 50.0
        
        return {
            "waveform": waveform,
            "hr": {"pred": hr_pred, "unc": torch.exp(0.5 * hr_out[:, 1])},
            "spo2": {"pred": spo2_pred, "unc": torch.exp(0.5 * spo2_out[:, 1])},
            "sbp": {"pred": sbp_pred, "unc": torch.exp(0.5 * bp_out[:, 1])},
            "dbp": {"pred": dbp_pred, "unc": torch.exp(0.5 * bp_out[:, 3])}
        }
```

#### 7. Statistical Evaluation Engine: Equivalence TOST & Cluster Bootstrap (B7)
```python
def compute_subject_cluster_bootstrap(errors_by_subject: dict, n_boot: int = 2000) -> dict:
    """Computes repeated-measures cluster bootstrap 95% CIs across subjects (Trap T8)."""
    sub_ids = list(errors_by_subject.keys())
    N = len(sub_ids)
    boot_maes = []
    
    for _ in range(n_boot):
        sample_subs = np.random.choice(sub_ids, size=N, replace=True)
        sample_errs = [e for sid in sample_subs for e in errors_by_subject[sid]]
        boot_maes.append(np.mean(sample_errs))
        
    return {
        "mean_mae": float(np.mean(boot_maes)),
        "ci_lower": float(np.percentile(boot_maes, 2.5)),
        "ci_upper": float(np.percentile(boot_maes, 97.5))
    }

def evaluate_c2_fairness_tost(strata_errors: dict, tau: float = 1.5, n_boot: int = 2000) -> dict:
    """Evaluates Two-One-Sided Equivalence Test (TOST) across ITA-1 to ITA-6 (B7)."""
    unweighted_mean = np.mean([np.mean(errs) for errs in strata_errors.values()])
    results = {}
    cohort_pass = True
    
    for stratum, errs in strata_errors.items():
        if len(errs) < 15: # Minimum sample size rule (B7)
            results[stratum] = {"status": "UNDERPOWERED", "n": len(errs), "ci_upper": None}
            cohort_pass = False
            continue
            
        boot_diffs = []
        for _ in range(n_boot):
            sample = np.random.choice(errs, size=len(errs), replace=True)
            boot_diffs.append(abs(np.mean(sample) - unweighted_mean))
        ci_upper = float(np.percentile(boot_diffs, 95.0))
        stratum_pass = ci_upper <= tau
        if not stratum_pass: cohort_pass = False
        results[stratum] = {"status": "PASS" if stratum_pass else "FAIL", "ci_upper": ci_upper, "tau": tau}
        
    return {"cohort_equivalence_pass": cohort_pass, "strata": results}
```

#### 8. Automated Submission Validator & .mat Exporter (HDF5 v7.3)
```python
import h5py

def export_submission_mat_v2(filename: str, facial_video: np.ndarray, roi_data: np.ndarray,
                             reference_signal: np.ndarray, processed_signal: np.ndarray,
                             metrics: dict, params: dict):
    """Exports official HDF5 (-v7.3) .mat file compliant with Section 1.2 & Trap T11."""
    assert facial_video.dtype == np.uint8, "facial_video must be uint8"
    with h5py.File(filename, 'w') as f:
        f.create_dataset('facial_video', data=facial_video, compression='gzip', compression_opts=4)
        f.create_dataset('roi_data', data=roi_data)
        f.create_dataset('reference_signal', data=reference_signal.astype(np.float64))
        f.create_dataset('processed_signal', data=processed_signal.astype(np.float64))
        
        g_m = f.create_group('metrics')
        g_m.attrs['MAE'] = float(metrics['MAE'])
        g_m.attrs['RMSE'] = float(metrics['RMSE'])
        g_m.attrs['Pearson_r'] = float(metrics['Pearson_r'])
        g_m.create_dataset('BlandAltman_LoA', data=np.array(metrics['BlandAltman_LoA'], dtype=np.float64))
        
        g_p = f.create_group('params')
        for k in ['camera_resolution', 'frame_rate', 'illuminance_lux', 'camera_to_subject_distance',
                  'subject_fitzpatrick_skin_type', 'subject_age', 'constraint_applied']:
            g_p.attrs[k] = params[k]
            
        g_e = g_p.create_group('extra')
        g_e.attrs['achieved_bitrate_kbps'] = float(params.get('achieved_bitrate_kbps', 500.0))
        g_e.attrs['measured_ita_deg'] = float(params.get('measured_ita_deg', 35.0))
        g_e.attrs['ita_stratum'] = str(params.get('ita_stratum', 'ITA-3'))
        g_e.attrs['c5_mode'] = str(params.get('c5_mode', 'ModeA'))
        g_e.attrs['coverage_fraction'] = float(params.get('coverage_fraction', 1.0))
    print(f"[PRE-FLIGHT PASS] Successfully written compliant submission: {filename}")
```

### 9.3 Cross-Platform Toolchain Mapping (MATLAB <-> Python Parity)

| Tool / Functionality | MATLAB Function / System Object [DOC] | Python / PyTorch Equivalent [TEAM-DEFINED] | Parity Caveat & Behavioral Mismatch (Traps T1, T3, T5, A7) |
| :--- | :--- | :--- | :--- |
| **Face Detection** | `faceDetector` / `vision.CascadeObjectDetector` | `mediapipe.FaceMesh` / `cv2.CascadeClassifier` | MATLAB `faceDetector` uses deep learning; OpenCV Haar cascade is faster but less accurate at low lux. |
| **Track Association**| `assignDetectionsToTracks` [DOC §2.2] | `scipy.optimize.linear_sum_assignment` | Exact Hungarian algorithm equivalence; ensure cost matrix uses bounding box IoU. |
| **Point Tracking** | `vision.PointTracker` (KLT) [DOC §2.2] | `cv2.calcOpticalFlowPyrLK` | Set identical pyramid levels ($L=3$) and window sizes ($15\times 15$) to match trajectory drift. |
| **Optical Flow** | `opticalFlowRAFT` [DOC §2.2] | `torchvision.models.optical_flow.raft_small` | Verify GPU memory limits on host; fallback to Farneback optical flow on edge Raspberry Pi. |
| **Shape Overlay** | `insertShape` [DOC §2.2] | `cv2.rectangle` / `cv2.fillPoly` | Ensure coordinate conventions match $[x, y, w, h]$ vs. $[x_1, y_1, x_2, y_2]$. |
| **Spatial Filtering**| `imfilter` [DOC §2.2] | `scipy.ndimage.correlate` / `cv2.filter2D` | Check boundary padding convention (`'replicate'` in MATLAB == `BORDER_REPLICATE` in OpenCV). |
| **Causal Filtering** | `designfilt` [DOC §2.3] | `scipy.signal.butter(output='sos')` | Use Second-Order Sections (`sos`) in Python to match numerical precision of MATLAB `designfilt`. |
| **Video Decoding** | `VideoReader` [DOC §2.2] | `cv2.VideoCapture` / `imageio.get_reader` | Ensure zero-indexed frame counts align with ground truth timestamp sequences. |
| **Video Encoding** | `VideoWriter` ('MPEG-4') [DOC §2.2] | `ffmpeg-python` (`libx264` CBR, A7d) | **Trap T3 & A7e:** Check `Quality` (0-100) vs. `VideoCompressionQuality` [VERIFY]; FFmpeg enforces true CBR. |
| **Color Conversion** | `rgb2lab` [DOC §2.2] | `cv2.cvtColor(RGB2LAB)` / `skimage` | **Trap T6:** `skimage` uses float ranges ($L \in [0, 100]$); OpenCV quantizes $L \in [0, 255]$. |
| **Image Resizing** | `imresize` (bicubic) [DOC §2.2] | `PIL.Image.BICUBIC` / `skimage(anti_aliasing=True)` | **Trap T5 & A7f:** MATLAB anti-aliases by default; raw OpenCV `INTER_CUBIC` does NOT! |
| **Precision Scaling**| `im2double` [DOC §2.2] | `img.astype(np.float64) / 255.0` | Identical conversion to range $[0.0, 1.0]$. |
| **Noise Injection** | `imnoise` ('poisson') [DOC §2.2] | `np.random.poisson(I * phi_full_well)` | **Trap T1:** MATLAB treats double as means scaled by $10^{12}$; must scale explicitly. |

---

## 10. Final Audit Appendices

### Appendix A: Assumptions and Unresolved-Dependency Register

| Tag | Category | Description | Impact if Invalidated | Action to Resolve | Owner Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[ASSUMPTION-01]**| Submission | Round 1 multi-condition submissions submitted as separate files (`Round1_Baseline_None.mat`, `Round1_LowLight_C1.mat`) [DOC §1.2] | Submission rejected for extra files | Email organizers (Item 1 in Appendix B) | Faculty Supervisor |
| **[ASSUMPTION-02]**| Reporting | Technical report page limits assumed as 4 pages for Round 1, 6 pages for Final Round | Disqualification for length violation | Check author kit upon portal launch | Faculty Supervisor |
| **[ASSUMPTION-03]**| Calibration | Nominal 50-lux gain assumed as 0.167 (300 to 50 lux under fixed exposure); $\Phi_{\text{full\_well}} = 5000\,\text{e}^-$ | Synthetic noise level miscalibrated | Calibrate against physical camera PTC curve | Undergrad 5 |
| **[ASSUMPTION-04]**| Evaluation | Primary evaluation window fixed at 10.0 s wall-clock across all frame rates (A1) | Frequency resolution confounded | Pre-register in `eval_protocol.yaml` | Undergrad 3 |
| **[VERIFY-01]** | Software | Confirm whether MATLAB's VideoWriter 'MPEG-4' property is `Quality` or `VideoCompressionQuality` (R3) | Script execution error in MATLAB | Check `propinfo(VideoWriter)` in R2024b/R2025a | Undergrad 1 |
| **[VERIFY-02]** | Architecture | Original PhysNet parameter count (~0.8M) vs. PhysNet-Lite (R3, A11) | Complexity claims invalidated | Audit PyTorch weight tensors of original repo | Undergrad 4 |
| **[VERIFY-03]** | Embedded | ARM NEON / MATLAB Coder support and INT8 performance for 3D convolutions (R3, B8) | Deployment failure on Pi 4/5 | Run Week 1 edge profiling spike | Undergrad 1 |
| **[VERIFY-04]** | Datasets | Confirm whether BP4D+ and V4V provide continuous BP labels and their licensing terms (R3, B2) | Data starvation for BP | Submit data use agreements on Day 2 | Faculty Supervisor |
| **[VERIFY-05]** | Experimental| Official Figure 1 appears to show a color tile chart on table; confirm if in-frame color reference (A4) | Calibration protocol non-compliance | Email organizers (Item 6 in Appendix B) | Faculty Supervisor |
| **[VERIFY-06]** | Clinical | ISO 81060-2 (BP: <= 5 +- 8 mmHg) and ISO 80601-2-61 (SpO2) status as aspirational benchmarks | Overclaiming medical compliance | State research prototype boundary in report | Tutor / Faculty |

### Appendix B: Document Discrepancies and Formal Organizer Clarification Email

#### The Six Clarification Inquiries
```text
To: vijayjeyakumar@ieee.org
Cc: sp-competitions@listserv.ieee.org, Jaqueline.Rash@ieee.org, lucas.thomaz@ieee.org
Subject: IEEE SP Cup 2027: Technical Clarifications on Data Organization & Submission Guidelines

Dear Dr. Vijay Jeyakumar and Competition Organizing Committee,

We are preparing our submission for the 2027 IEEE Signal Processing Cup ("FaceVitals"). To ensure rigorous compliance with your evaluation protocols, our team respectfully requests clarification on the following six technical items:

1. Round 1 Multi-Condition File Artifacts: Task 1 requires demonstrating HR extraction under both controlled and uncontrolled conditions. In Section 1.2, only 'Round1_Baseline_None.mat' is explicitly itemized. Should teams submit multiple .mat files for Round 1 (e.g., 'Round1_Baseline_None.mat' alongside 'Round1_LowLight_C1.mat' and 'Round1_LowResolution_C4.mat'), or package all conditions into a single container?
2. Directory Structure Reconciliation: Section 2.3 outlines three competition rounds, whereas Section 1.2 lists four directories (/Round1_Baseline/, /Round2_HR_SpO2/, /Round3_Constrained/, /Round4_FullSystem_Clinical/). Could you confirm whether Round 3 final deliverables should be placed in /Round3_Constrained/ or /Round4_FullSystem_Clinical/?
3. Scoring Criteria Count: Section 5.2 notes that teams will be scored on "five criteria" but tabulates six equally weighted criteria (Innovation, Val Performance, Test Performance, Complexity, Report, Presentation). We assume all six criteria in the table are active and weighted at 16.67% each; please confirm.
4. Condition Tag for Low Bandwidth (C5): Section 3 specifies condition tags 'Controlled', 'Hospital', 'LowLight', 'LowRes'. What is the official tag for Low-Bandwidth/Frame-Rate captures under C5 (e.g., 'LowBandwidth' or 'LowFPS')?
5. Mode A On-Device Trace Streaming for C5: Constraint C5 specifies an "on-device or low-bandwidth offline mode". Does an edge implementation that extracts ROI color traces on-device at 30 fps and transmits 1D traces (~4.3 kbps, well below 500 kbps) satisfy C5, or must the transmission strictly be a compressed video bitstream?
6. Reference Color Tile Chart: Figure 1 appears to depict a standard color tile chart positioned on the experimental table. Is this chart intended as an in-frame radiometric color reference, and will its spectral reflectance values be published in the reference protocol?

Thank you for your guidance and dedication to this exceptional initiative.

Sincerely,
Faculty Supervisor & Student Team, IEEE SPS Student Branch Chapter
```

### Appendix C: Ethically and Clinically Sensitive Claims Register

```text
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
│    without NIR Sensing         │ surrogate. Due to broad overlapping spectral bands, variable tissue   │
│                                │ penetration, and skin-tone optical absorption, readings outside the   │
│                                │ 90–100% saturation range cannot be verified without direct oximetry."│
├────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ 3. Cuffless Blood Pressure     │ "Cuffless camera-based BP estimation relies on pulse wave morphology  │
│    Estimation                  │ and non-demographic priors. It is highly susceptible to acute vascular│
│                                │ changes and is unvalidated in arrhythmia, shock, or hypotensive crisis│
├────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ 4. Vitiligo & Dermatological   │ "Synthetic depigmentation testing demonstrates algorithmic resilience │
│    Generalization Claims       │ to local reflectance gradients; however, performance on clinical      │
│                                │ vitiligo cannot be claimed without extensive multi-center data."      │
└────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

### Appendix D: Minimal Extension Hooks for C3, C6, and C7
* **C3 Multi-Person Isolation Hook:** Bounding boxes from face detection pass through Hungarian association (`assignDetectionsToTracks` [DOC §2.2]). A locked target ID maintains tracking via Kalman velocity filtering, discarding bystander coordinates.
* **C6 Aging-Skin Micro-Shadow Hook:** An edge-preserving bilateral spatial prefilter smooths localized wrinkle micro-shadows prior to spatial averaging, preventing wrinkle amplitude modulation from folding into cardiac frequencies [DOC §2.2].
* **C7 Masked-Face Dynamic ROI Re-weighting:** A landmark confidence check evaluates lower-face occlusion. If lower cheek landmark confidence drops below 0.3 (indicating a surgical mask [DOC §2.2]), the extraction pipeline redistributes 100% of the signal weight to the forehead and periocular ROIs.

### Appendix E: Evidence-Based Self-Audit & Compliance Verification (A13)

| Requirement / Constraint | Audit Status | Evidence & Verification Location | Open Action / Dependency |
| :--- | :---: | :--- | :--- |
| **C1 Low Light (<= 50 lux)** | **CONFIRMED** | Addressed in §1.1, §2.2, §3.4 (Simulator v2), §4.2, §5.1, §6.3, §7 (R1), §9.2 | Calibrate 50-lux gain on real camera [ASSUMPTION-03] |
| **C2 Skin Tone (ITA-1 to ITA-6)**| **CONFIRMED** | Addressed in §1.1, §2.2, §3.2, §4.4, §5.1, §6.2 (TOST), §7 (R2), §9.2 | Obtain Tier B dataset agreement [OPEN: 8 Nov gate] |
| **C4 Low Resolution (QVGA)** | **CONFIRMED** | Addressed in §1.1, §2.2, §3.4 (Anti-aliased resize), §5.1, §6.3, §7 (R3), §9.2 | Verified on public UBFC video data |
| **C5 Low Bandwidth / FPS** | **CONFIRMED** | Addressed in §1.1, §2.2, §3.4 (Modes A & B), §5.1, §6.3, §7 (R4), §9.2 | Confirm Mode A acceptance with organizers (App. B) |
| **Mode A (Trace Streaming)** | **CONFIRMED** | Specified in §2.1, §2.2, §3.4, §6.3; bandwidth arithmetic ($4.32\,\text{kbps}$) verified | Confirm Mode A satisfies C5 with organizers [OPEN] |
| **Mode B (Degraded Video)** | **CONFIRMED** | Specified in §2.1, §2.2, §3.4, §6.3; flicker aliasing worked example in §3.4 | FFmpeg libx264 rate-control script implemented |
| **Window Length Resolution** | **CONFIRMED** | $\Delta f = 1/T$ table in §2.2; fixed 10.0 s primary window enforced in §2.2/§6.3 | Erroneous constant-frame-count claims eliminated |
| **PTT Claim Integrity** | **CONFIRMED** | Sub-frame delay limitations stated in §1.2 & §2.4; PTT removed from §2.1 BP box | Dogmatic impossibility claims replaced with SNR bounds |
| **Anti-Aliasing Integrity** | **CONFIRMED** | Replaced with frame skipping + average-then-skip + flicker example in §3.4 & §7 | Flawed post-decimation prefilter claims eliminated |
| **ITA Stratification Naming** | **CONFIRMED** | Designated `ITA-1` to `ITA-6` across §1.1, §3.2, §4.4, §6.2, §9.1, §9.2 | Zero instances of "Fitzpatrick Type I–VI" as ITA labels |
| **BP Demographic Inputs** | **CONFIRMED** | Formula in §2.4 and code in §9.2 use strictly `Age, BMI`; demographic fallback labeled | Zero instances of ITA in BP input features |
| **Simulator v2 Quantization** | **CONFIRMED** | 8-bit quantization, auto-exposure modes, and order ablation in §3.4 & §9.2 | Calibrate against physical validation clips [OPEN] |
| **Tier A HD Baseline Absence**| **CONFIRMED** | Acknowledged in §3.1 & §5.1 that UBFC is VGA; HD baseline requires Tier B/C | Tier C IRB collection initiated [OPEN: Day 6] |
| **Weak Loss for Scalar HR** | **CONFIRMED** | $\mathcal{L}_{\text{HR-scalar}}$ formulated in §4.2 for `rppg_biases` | Code implemented in §9.2 |
| **Group-DRO Limitations** | **CONFIRMED** | Convexity limits, sample-size regularization, and ERM ablation in §4.2 & §7 | Unbounded optimization claims eliminated |
| **Pre-Implementation Targets**| **CONFIRMED** | Every parameter count, FLOP count, latency, and accuracy target carries `[MEASURE]` | Run edge profiling spike in Weeks 1–2 [OPEN] |
| **Scope & Split Consistency**| **CONFIRMED** | R1 scope reconciled in §1.3, §5.1, §5.2, §8.4; 5-fold CV on dev cohort in §3.2 | Pre-register in `eval_protocol.yaml` |
| **Valid Resources (No MIMIC)**| **CONFIRMED** | Replaced in §2.4 & §7 with PulseDB contact PPG pretraining domain-gap caveat | Invalid dataset references eliminated |
| **Six .mat File Variables** | **CONFIRMED** | Listed in §5.3 and code exporter in §9.2 | Format verified by unit test `test_mat_format.py` |
| **Four Official Metrics** | **CONFIRMED** | MAE, RMSE, Pearson $r$, repeated-measures Bland-Altman LoA in §5.3, §6.1, §9.2 | Bootstrap code implemented in §9.2 |
| **First-8-Days Action List** | **CONFIRMED** | Exactly Day 1 to Day 8 (17–24 Oct) in §8.2, ending before 25 Oct deadline | Execute registration by 24 Oct [OPEN] |
