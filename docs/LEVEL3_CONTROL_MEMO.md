# W-bit Level 3 Control Stack Memo

**DATE:** March 15, 2026
**STATUS:** Pre-Silicon Physics-Model Verification Complete

This memo formalizes the Level 3 runtime adaptation rules for the W-bit fabric. It translates the qualitative insights of the Level 3 Ascension Report into concrete mathematical conditions, hysteresis laws, and failure envelopes required for hardware implementation.

---

## 1. State Variables & Physical Definitions

- $R_{max}$: Maximum physical addressable states per cell (e.g., 9).
- $R_{eff}(t)$: Effective active states scheduled by the OS at time $t$ ($R_{eff} \in \{3, 5, 7, 9\}$).
- $V_m(i, t)$: Analog membrane potential (voltage) of cell $i$ at time $t$.
- $\tau_m$: Capacitive leak time constant.
- $N_{est}(t)$: Running estimate of environmental/thermal noise.
- $S_{peak}$: Nominal peak signal drive voltage from local crossbar interaction.
- $\text{SNR}(t)$: Signal-to-Noise Ratio, approximated as $S_{peak} / N_{est}(t)$.

---

## 2. Mode Transition Conditions (The Control Law)

The Adaptive OS continuously monitors the local SNR. It transitions the operating regime according to the following thresholds:

- **$R_{eff} = 9$ (High Precision):** $\text{SNR} > 2.0$
- **$R_{eff} = 7$ (Moderate Precision):** $1.5 < \text{SNR} \le 2.0$
- **$R_{eff} = 5$ ($\phi$-transition):** $1.0 < \text{SNR} \le 1.5$
- **$R_{eff} = 3$ ($\phi$-mode, Max Robustness):** $\text{SNR} \le 1.0$

### Mode Actions:
When transitioning $R_{eff} \rightarrow r$:
1. **ADC Read Margins:** Reconfigure the comparators to group the $R_{max}$ continuous voltage range into $r$ bins.
2. **$\beta$-Bias Injection:** For the central state $c$ of each active bin, apply a constant pre-charge current:
   $$ I_{\beta, c} = \beta_{mult} \cdot \ln(R_{max} / r) $$
   *(Verified $\beta_{mult} \approx 1.127$ for optimal mass restoration without flattening the gradient).*
3. **Landscape Smoothing:** The OS *must* ensure that the programmed compatibility matrix ($\Theta$) is low-rank (WZMA structured). If the underlying logic requires a rigid/sparse matrix, dropping to $\phi$-mode will cause catastrophic failure (as verified by the Base Matrix collapse).

---

## 3. Hysteresis & Thrashing Prevention

Rapid oscillation between $R_{eff}$ modes (thrashing) will destroy temporal integration and consume excessive switching energy.

**Transition Hysteresis Rule:**
- **Downgrade (e.g., $9 \rightarrow 3$):** Immediate. Triggered if SNR falls below the target threshold for $> 3 \tau_m$ (to avoid overreacting to single-step bursts).
- **Upgrade (e.g., $3 \rightarrow 9$):** Delayed. The SNR must exceed the upgrade threshold by a safety margin of $+0.5$ and remain stable for $> 100 \tau_m$.

---

## 4. Local Update Equations (Temporal Integration)

At any given sub-step $dt$, the physical cell integrates input according to the analytical LIF equation:
$$ V_m(t+dt) = V_m(t)e^{-dt/\tau_m} + I_{total} \tau_m (1 - e^{-dt/\tau_m}) $$

Where $I_{total}$ is the sum of:
- Crossbar current: $\sum_j G_{ij} \cdot S_j$
- $\beta$-Bias pre-charge (if applicable)
- Environmental noise (Gaussian or burst)

If $V_m(t+dt) > V_{threshold}$, a logical spike (or continuous analog read) is registered and the capacitor discharges ($V_m \leftarrow V_m \pmod{V_{threshold}}$).

---

## 5. Failure Envelopes (Simulation Derived)

- **Gaussian Catastrophe Limit:** $\phi$-mode ($R=3$) maintains $>96\%$ logical fidelity up to $\text{SNR} = 0.6$. Below $\text{SNR} = 0.5$ (Noise $\ge 2 \times$ Signal), accuracy degrades linearly to random chance (33%).
- **Brittle Matrix Collapse:** If $R_{eff} = 3$ is triggered on a full-rank / rigid diagonal compatibility matrix, fidelity drops to $<40\%$. The low-rank (WZMA) geometry is an absolute physical prerequisite for $\phi$-mode survival.
- **High-Precision Collapse:** $R_{eff} = 9$ logic completely disintegrates ($<60\%$ accuracy) if $\text{SNR} < 1.0$.

---

## 6. Hardware Validation Ladder (Pending)

To graduate from pre-silicon simulation to physical validation, the following device-level characterizations are required:

1. **IR Drop / Line Parasitics:** Quantify voltage degradation across the physical crossbar array. Ensure the $\beta$-bias current does not saturate neighboring unselected wordlines.
2. **Comparator Offset Drift:** Validate the ADC margin widening. Ensure threshold voltages ($V_{ref}$) do not drift into the $\phi$-mode active centers under sustained thermal heating.
3. **Burst Noise Recovery:** Measure the physical latency for $V_m$ to stabilize back into the correct WZMA energy basin after a high-voltage non-Gaussian transient shock.
4. **Energy Profiling:** Measure the $fJ/op$ (femtojoules per operation) cost of maintaining $\tau$-bit capacitive integration vs. instantaneous reading. 
   *Simulation Estimates:*
   - **High-Precision ($R=9$, Instantaneous):** ~40 fJ/op (Highly efficient, used when SNR > 2.0).
   - **Survival Mode ($R=3$, Temporal Integration + $\beta$-bias):** ~516 fJ/op (A 13x energy penalty).
   This confirms that Level 3 is a true emergency survival mechanism. It deliberately sacrifices energy efficiency (burning 13x more power per operation via temporal integration and DC bias injection) to guarantee logical fidelity during catastrophic environmental interference. The OS must actively roll back to High-Precision mode as soon as the noise clears to prevent thermal overload and preserve the TDP budget.