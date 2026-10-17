# FaceVitals: Engineering Directives & Coding Standards (Rules)

### 1. Mandatory Zero-Hardcoding Contract
1. **NO Hardcoded Output Arrays:** Signal arrays, metrics, and parameters must be computed dynamically through real signal processing passes. Never commit mock arrays pretending to be real extracted vital signs.
2. **NO Hardcoded File Paths:** All paths must resolve relative to repository root or configurable command-line arguments.
3. **NO Arbitrary Pixel Bounding Boxes:** Never hardcode $60\times 30$ or $25\times 25$ pixel boxes. All ROIs must be proportional fractions of the detected face bounding box, scaling gracefully from $1080\text{p}$ down to $240\text{p}$ (QVGA).
4. **NO Hardcoded Demographic Cheats for Blood Pressure:** Pulse wave morphology, ejection time, and stiffness index must derive from true photoplethysmogram waveforms, not demographic lookup tables.

---

### 2. Numerical Stability & Type Safety
1. **Typing Mandate:** Every Python function and class must include complete type annotations (`typing.Dict`, `typing.Tuple`, `typing.Optional`, `numpy.ndarray`).
2. **Filter Numerical Stability:** Always use Second-Order Sections (`sos`) representation for IIR digital filters (`scipy.signal.sosfilt`), avoiding transfer function polynomials (`b, a`) that suffer from numerical pole collapse at high sampling rates.
3. **Division by Zero Protection:** Add $\epsilon = 10^{-7}$ guards on all standard deviation normalizations, SNR calculations, and color ratios.
4. **No NaNs in Graded Submission Files:** When signal quality drops below acceptable thresholds, the system records the best-effort estimate paired with a low quality flag (`SQI < 0.40`), never emitting unhandled `NaN` entries that cause MATLAB evaluation crashes.

---

### 3. MathWorks MATLAB Compatibility Rules
1. All `.mat` files generated for submission must use MATLAB-compliant formats (`scipy.io.savemat` with `do_compression=True` or HDF5 `-v7.3`).
2. Array orientations must align with MATLAB column-major conventions where applicable.
3. The native MATLAB entry point `process_round1.m` must execute standalone without external dependencies, relying strictly on standard MathWorks toolboxes.
