# FaceVitals: Scientific Visualization & Reporting Contract (Design)

### 1. IEEE Publication Plotting Standards
1. **Typography:** Consistent font hierarchy using sans-serif fonts (Helvetica, Arial, or DejaVu Sans) matching IEEE conference templates (8 pt tick labels, 10 pt axis labels, 12 pt titles).
2. **Color Palette:** High-contrast, colorblind-friendly palettes (viridis, cividis, or colorbrewer Dark2). Melanin / ITA plots must use continuous lightness maps rather than arbitrary categorical hues.
3. **Bland–Altman Plots:**
   - Scatter points plotted as $(y_i + \hat{y}_i)/2$ on x-axis vs $(y_i - \hat{y}_i)$ on y-axis.
   - Mean bias line (solid dark line) and $95\%$ Limits of Agreement ($\bar{d} \pm 1.96 s_d$) as dashed lines.
   - Confidence intervals on LoA derived via subject-level cluster bootstrap (shaded bands).
4. **Feasibility Maps:** 2D surface plots displaying CRLB theoretical error contours vs. empirical model error across illuminance (lux) and frame rate (fps).
