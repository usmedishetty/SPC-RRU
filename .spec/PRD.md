# FaceVitals: Product Requirements Document (PRD)
## 2027 IEEE Signal Processing Cup (ICASSP 2027, New Orleans)

### 1. Executive Summary & Problem Statement
Contactless physiological monitoring using standard consumer RGB cameras enables remote vital sign assessment without dedicated medical sensors (pulse oximeters, blood pressure cuffs, ECG leads). The 2027 IEEE Signal Processing Cup ("FaceVitals") challenges participants to extract three vital physiological parameters:
1. **Heart Rate (HR)** [Round 1 Baseline & Beyond]
2. **Peripheral Oxygen Saturation ($\text{SpO}_2$)** [Round 2 Joint Estimation]
3. **Blood Pressure (Systolic & Diastolic)** [Round 3 Clinical Validation & Edge Deployment]

The system must operate purely from standard RGB facial video (no NIR, thermal, or contact sensors) across both offline (recorded video) and online (live-stream) modes, and generalize across real-world deployment constraints.

---

### 2. Quantified Deployment Constraints (Section 2.2)

| Constraint Tag | Operating Specification | Physical Phenomenon & Success Criteria |
| :--- | :--- | :--- |
| **C1: Low-Light** | Ambient illuminance $\le 50\,\text{lux}$ | Photon shot noise and sensor read noise dominant; must not fail or lose cardiac frequency. |
| **C2: Skin Tone** | Fitzpatrick types I–VI, objective ITA stratification | Melanin attenuation of shorter wavelengths; per-type error must not deviate beyond tolerance from population mean. Explicit handling of vitiligo depigmentation. |
| **C3: Multi-Person** | Multi-subject video frames | Automatically detect, isolate, and continuously track designated subject; reject bystander pulse signals. |
| **C4: Low Resolution** | $\le 640\times 480$ (VGA), stress at $\le 320\times 240$ (QVGA) | System must degrade gracefully, maintaining sub-pixel tracking and ROI trace integration. |
| **C5: Low Bandwidth** | Bitrate $\le 500\,\text{kbps}$, frame rate $\le 10\,\text{fps}$ | Frame decimation near Nyquist limit ($\approx 6\,\text{fps}$) and MPEG-4 block compression artifacts. |
| **C6: Aging Skin** | Elderly subjects (ages 60–70+) | Wrinkle micro-shadowing and reduced elasticity; accuracy reported separately for subgroup. |
| **C7: Masked Face** | $\ge 40\%$ occlusion (surgical mask) | Identifies lower-face occlusion and dynamically re-weights 100% to exposed forehead/periocular ROIs. |

---

### 3. Competition Phases & Deliverables

#### Round 1: Baseline Heart Rate Extraction (Deadline: 30 November 2026)
* Extract Heart Rate (HR) in bpm from controlled ($\ge 300\,\text{lux}$, $720\text{p}$, $\ge 30\,\text{fps}$) and uncontrolled video.
* Explicitly address and evaluate against at least two constraints (C1–C7).
* Submission artifacts:
  - Technical report in IEEE conference format.
  - `process_round1.m` / Python processing script.
  - `Round1_Baseline_None.mat` and constrained `.mat` containers.
  - Synchronized `reference_signal.csv` and `processed_output.csv`.

#### Round 2: Two-Parameter Extraction (HR + $\text{SpO}_2$) (Deadline: 16 January 2027)
* Joint estimation of HR and $\text{SpO}_2$ from RGB video.
* Quantitative comparison against clinical pulse oximetry.

#### Round 3: Three-Parameter Clinical System & Live Edge Demo (May 2027, ICASSP)
* Simultaneous estimation of HR, $\text{SpO}_2$, SBP, and DBP.
* Real-time demonstration on target edge platform (Raspberry Pi 4/5) under $\ge 2$ stress constraints.

---

### 4. Evaluation Metrics (Section 6)
1. **Mean Absolute Error (MAE):** $\frac{1}{N}\sum |y_i - \hat{y}_i|$
2. **Root Mean Square Error (RMSE):** $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$
3. **Pearson Correlation ($r$):** Covariance over product of standard deviations.
4. **Bland–Altman 95% Limits of Agreement (LoA):** $\bar{d} \pm 1.96 \cdot s_d$ with repeated-measures cluster bootstrap.
5. **C2 Fairness Criterion:** Two-One-Sided Equivalence Test (TOST) bound and Worst-Stratum MAE.
