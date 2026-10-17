# FaceVitals: Application & Pipeline State Flow (AppFlow)

### 1. State Machine: Pipeline Execution Modes

```text
       ┌──────────────────────┐
       │     IDLE STATE       │
       └──────────┬───────────┘
                  │ Input: Video file (MP4/AVI) or Camera Device Stream
                  ▼
       ┌──────────────────────┐
       │  INITIALIZATION      │ ──► Parse video header: fps, resolution, total_frames
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ FRAME INGESTION LOOP │ ◄─────────────────────────────────────────────┐
       └──────────┬───────────┘                                               │
                  │ Frame t                                                   │
                  ▼                                                           │
       ┌──────────────────────┐                                               │
       │ PREPROCESSING & GATING│                                              │
       │ - Low lux Retinex    │                                               │
       │ - Relative ROI track │                                               │
       │ - Mask occlusion chk │                                               │
       └──────────┬───────────┘                                               │
                  │ Track OK?                                                 │
          ┌───────┴───────┐                                                   │
     YES  │               │ NO (Lost / Severe Jitter)                         │
          ▼               ▼                                                   │
 ┌─────────────────┐ ┌────────────────────────┐                               │
 │ Extract 3 ROIs: │ │ Fallback to center     │                               │
 │ Spatial Mean    │ │ face bounding box      │                               │
 └────────┬────────┘ └───────────┬────────────┘                               │
          │                      │                                            │
          └───────────────┬──────┘                                            │
                          ▼                                                   │
               ┌──────────────────────┐                                       │
               │ SLIDING BUFFER (W s) │ ── (Buffers next frame) ──────────────┘
               └──────────┬───────────┘
                          │ Buffer Full (e.g., T >= 5.0 s)?
                          ▼
               ┌──────────────────────┐
               │ CLASSICAL / TCN PASS │
               │ - POS projection     │
               │ - 4th-order SOS IIR  │
               │ - Harmonic comb peak │
               │ - Tri-SQI evaluation │
               └──────────┬───────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │ CONVERSIONS & LOGGING│
               │ - Output HR, SpO2, BP│
               │ - Compute CRLB bound │
               │ - Append CSV stream  │
               └──────────┬───────────┘
                          │ Loop until EOF
                          ▼
               ┌──────────────────────┐
               │ FINAL EXPORT & .MAT  │
               │ - Calculate metrics  │
               │ - Save validated MAT │
               └──────────────────────┘
```
