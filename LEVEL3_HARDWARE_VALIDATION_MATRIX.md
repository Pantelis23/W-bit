# W-bit Level 3: Hardware Validation Matrix

**DATE:** March 16, 2026
**STATUS:** Validation Ladder Definition

This document outlines the multidimensional failure surfaces and the empirical testing required to elevate the Level 3 Architecture from "Simulation-Verified" to "Hardware-Proven."

---

## 1. Validation Matrix

| Test Dimension | Parameter Sweep | Simulator Status | Bench Status | Pass Criterion |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian Catastrophe** | SNR: $3.0 \rightarrow 0.5$ | ✅ Verified (96.7% @ 0.6) | ⏳ Pending | $\phi$-mode maintains $>95\%$ fidelity down to SNR 0.6. |
| **Burst Transient (Shock)** | $\Delta V$: $+10V \rightarrow +50V$ | ✅ Verified (99.1% @ 20V) | ⏳ Pending | Shock Detector floats lines; $<1\%$ state corruption per event. |
| **Low-Freq Drift (Heating)** | $f_{drift} < 1$ Hz, $\Delta V = \pm 5V$ | ✅ Verified (99.8%) | ⏳ Pending | WZMA slopes cleanly absorb drift without triggering false margin spikes. |
| **Asymmetric Offset** | $V_{offset}: -1V \rightarrow -5V$ | ✅ Verified (99.9% @ -4V) | ⏳ Pending | $\phi$-mode wide ADC margins absorb systematic offset. |
| **Thermal / Temperature** | $T_{ambient}: 20^\circ C \rightarrow 120^\circ C$ | ❌ Unmodeled | ⏳ Pending | Leakage ($\tau_{leak}$) remains bounded; system does not enter thermal runaway. |
| **IR Drop (Parasitics)** | Array size: $16 \times 16 \rightarrow 256 \times 256$ | ❌ Unmodeled | ⏳ Pending | $\beta$-bias injected at row driver maintains $>80\%$ amplitude at furthest column. |
| **Cross-Talk (Saturation)** | $\Delta V_{neighbor}: 0 \rightarrow V_{max}$ | ❌ Unmodeled | ⏳ Pending | $\beta$-bias applied to active wordlines does not flip adjacent inactive rows. |
| **Duration Under Stress** | Time: $1\mu s \rightarrow 1$ hour | ✅ Verified (1B cycles) | ⏳ Pending | No progressive hysteresis degradation over sustained continuous use. |
| **Energy Overhead** | $fJ/op$ profiling | ✅ Modeled (13x penalty) | ⏳ Pending | Measured $fJ/op$ aligns with simulation; TDP budget is not exceeded. |

---

## 2. Failure Envelopes (Multidimensional)

Physical validation must map the boundaries where Level 3 intervention is no longer sufficient.

### 2.1 Temperature vs. $\tau$-Leak Envelope
As ambient temperature increases, the natural physical leak rate of the capacitors increases. 
- **Risk:** If $\tau_{leak}$ becomes faster than the integration required to average out the noise, the $\tau$-bit fails.
- **Envelope Mapping:** Plot Survival % across an axis of `SNR` vs. `T_ambient`.

### 2.2 Array Size vs. $\beta$-Bias Delivery
The $\beta$-bias is a physical DC current injected from the edge of the tile.
- **Risk:** In massive tiles ($256 \times 256$), parasitic wire resistance (IR drop) will weaken the bias at the far end of the array, causing those specific cells to fail while the near-edge cells survive.
- **Envelope Mapping:** Plot Survival % across an axis of `SNR` vs. `Column Index`. Identify the maximum safe tile dimension for $\phi$-mode operation.

### 2.3 Comparator Offset vs. Matrix Rigidity
Manufacturing defects can cause ADC comparators to drift permanently.
- **Risk:** If the WZMA matrix is not sufficiently low-rank (i.e., the basins are too narrow), a permanent comparator offset will push the physical peak outside the read margin.
- **Envelope Mapping:** Plot Survival % across `WZMA Rank` vs. `V_offset`.

---
## 3. Progression Criteria

The Level 3 architecture will be considered **Hardware-Proven** when:
1. A physical $64 \times 64$ crossbar tile can autonomously transition $9 \rightarrow 3 \rightarrow 9$ under an externally applied noise profile (simulating SNR drop).
2. The empirical survival rate matches the simulation trajectory (within $5\%$ variance).
3. The total measured energy draw during Survival Mode matches the 13x penalty model without triggering localized thermal runaway.