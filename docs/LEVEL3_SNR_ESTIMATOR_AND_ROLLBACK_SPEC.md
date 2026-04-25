# W-bit Level 3: SNR Estimator & Rollback Specification

**DATE:** March 16, 2026
**STATUS:** Formal Control Stack Definition

This document defines the exact physical and algorithmic mechanisms for local Noise Estimation ($N_{est}$), the fast shock-detection path, spatial mode granularity, and the precise rollback laws required to safely exit Level 3 Survival Mode.

---

## 1. Spatial Granularity & Boundary Rules

Level 3 self-modification is not applied per-cell (which would cause chaotic data routing) nor globally (which would waste massive thermal energy).

- **Granularity:** Mode switching ($R_{eff}$) is enforced at the **Tile Level** (e.g., $64 \times 64$ crossbar array).
- **Boundary Rules (Mixed-Mode Interaction):**
  - When data routes from an $R_{eff} = 9$ tile to an $R_{eff} = 3$ tile, the receiving tile's wide ADC margins automatically bin the high-precision signal into the appropriate $\phi$-mode bucket.
  - When data routes from an $R_{eff} = 3$ tile to an $R_{eff} = 9$ tile, the signal is physically restored. The $R=3$ tile emits a pure analog voltage corresponding exactly to the center of its bin (e.g., State 1, 4, or 7). The receiving $R=9$ tile accepts this as a valid, precise state.
  - **Isolation:** $\beta$-bias pre-charge is localized to the specific active wordlines within the tile. Substrate isolation trenches must physically decouple neighboring tiles to prevent bias current from saturating adjacent High-Precision tiles.

---

## 2. SNR Estimator ($N_{est}$)

The OS cannot pause computation to calculate standard deviations. Noise estimation must be continuous, cheap, and analog-derived.

### 2.1 The Two-Channel Detector
Each tile maintains two parallel hardware estimators:
1. **Slow Drift Estimator (The Integration Channel)**
2. **Fast Shock Detector (The Transient Channel)**

### 2.2 Slow Drift Estimator ($N_{est}$)
Measures sustained thermal interference or supply ripple.
- **Physical Mechanism:** A dedicated dummy column in the tile acts as the noise sentinel. It is driven with a constant DC voltage $V_{ref}$.
- **Update Equation:** The variance of the output current $I_{out}$ is integrated using a leaky Exponential Moving Average (EMA):
  $$ N_{est}(t) = (1 - \alpha) \cdot N_{est}(t-dt) + \alpha \cdot |I_{out}(t) - I_{ref}| $$
- **Window:** $\alpha$ is tuned so the half-life of the EMA is $100 \tau_m$.
- **Trigger:** If $S_{peak} / N_{est}(t)$ drops below the defined thresholds for $> 3 \tau_m$, initiate staged downgrade.

### 2.3 Fast Shock Detector (Burst Noise)
Measures cosmic rays, electrostatic discharge (ESD), or sudden massive load spikes.
- **Physical Mechanism:** A high-pass filter attached to the global bitline.
- **Trigger:** If $dV/dt$ exceeds $V_{shock\_limit}$ (e.g., a massive 20V spike in $1$ ns).
- **Action (The Shock Path):**
  - **Immediate Halt:** Instantly disconnect crossbar inputs (Float the bitlines) to prevent the burst from writing corrupted data into the capacitors.
  - **Bleed Off:** Wait $5 \tau_{leak}$ for the transient spike to dissipate from the substrate.
  - **Resume:** Reconnect lines. Do *not* change $R_{eff}$ unless the Slow Drift Estimator subsequently confirms the baseline noise floor has permanently risen.

---

## 3. The Rollback Law (Exit Path)

Exiting Survival Mode ($3 \rightarrow 9$) is more dangerous than entering it. An abrupt exit can induce electrical ringing (Lenz's Law) and corrupt the precise bins. Rollback must be hysteretic and staged.

### 3.1 Exit Condition
The Slow Drift Estimator must report $\text{SNR} > \text{Threshold} + 0.5$ continuously for $> 100 \tau_m$.

### 3.2 Staged Recovery (The Step-Up Sequence)
The OS does not jump directly from $3 \rightarrow 9$. It walks the ladder: $3 \rightarrow 5 \rightarrow 7 \rightarrow 9$. Each step takes $10 \tau_m$ to allow voltages to settle.

### 3.3 The Decay Schedules
When stepping from $R_{eff} = r_{current}$ to $r_{next}$ (where $r_{next} > r_{current}$):

1. **$\beta$-Bias Decay:**
   - Do not cut the current instantly.
   - Ramp down $I_{\beta}$ linearly over $5 \tau_m$ before activating the new ADC margins. This prevents inductive voltage spikes on the wordlines.
   
2. **Temporal Window ($\tau$) Decay:**
   - The integration window shrinks proportionally to the mode. 
   - $R=3$ uses $\tau = 50$ ns. $R=9$ uses instantaneous read.
   - As $R_{eff}$ increases, the OS gradually shortens the read-enable strobe.

3. **State Recentering:**
   - Before handing control back to instantaneous $R=9$ logic, the tile forces one final "clean" $\tau$-bit write cycle to ensure all capacitors are holding exact, centered voltages, purging any residual "slop" from the wide $\phi$-mode basins.