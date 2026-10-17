# FaceVitals: Data Models & Submission Schemas (Schema)

### 1. Mandatory `.mat` File Schema (Call for Proposals §1.2)

Every submission file (e.g., `Round1_Baseline_None.mat`) must contain the following 6 variables:

| Variable Name | MATLAB Type | Dimensions | Physical Description |
| :--- | :--- | :--- | :--- |
| `facial_video` | `uint8` | $[T \times H \times W \times 3]$ | Raw RGB video frames of the evaluated subject |
| `roi_data` | `double` / `int32` | $[T \times 3 \times 4]$ | Bounding boxes $[x, y, w, h]$ for Forehead, Left Cheek, Right Cheek per frame |
| `reference_signal` | `double` | $[T \times 1]$ | Synchronized ground-truth physiological signal (HR in bpm) |
| `processed_signal` | `double` | $[T \times 1]$ | Algorithmic output signal matching reference temporal grid |
| `metrics` | `struct` | Scalar fields | Objective evaluation results: `MAE`, `RMSE`, `Pearson_r`, `BlandAltman_LoA` $[2 \times 1]$ |
| `params` | `struct` | Scalar fields | Metadata: `camera_resolution`, `frame_rate`, `illuminance_lux`, `camera_to_subject_distance`, `subject_fitzpatrick_skin_type`, `subject_age`, `constraint_applied` |

---

### 2. CSV Output Schemas

#### `reference_signal.csv`
```csv
timestamp_ms,reference_hr_bpm,reference_spo2_pct,reference_sbp_mmhg,reference_dbp_mmhg
0.0,72.4,98.2,118.0,76.5
33.3,72.5,98.2,118.2,76.6
```

#### `processed_output.csv`
```csv
timestamp_ms,estimated_hr_bpm,estimated_spo2_pct,estimated_sbp_mmhg,estimated_dbp_mmhg,sqi_score,abstain_flag
0.0,71.9,98.0,119.1,77.2,0.89,0
33.3,72.1,98.0,119.0,77.1,0.91,0
```
