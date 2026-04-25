# W-bit Level 3: Bench Testing & Physical Characterization Plan

**DATE:** March 16, 2026
**STATUS:** Operational Test Plan (Pending Bench Execution)

This document translates the theoretical Validation Matrix into concrete physical bench experiments. It focuses exclusively on the "Fabric-Scale Ugliness" that simulation abstracts away: spatial non-uniformity, parasitic impedance, and asymmetric thermal loading.

---

## 1. Estimator Fidelity Test (The "Sentinel" Validation)

**Objective:** Prove that the dummy-column noise sentinel accurately proxies the active crossbar area under highly asymmetrical stress.
**Hypothesis:** If a tile is subjected to a steep spatial thermal gradient, a single edge-mounted sentinel will fail to estimate the $N_{est}$ of the opposite corner, leading to premature or delayed Level 3 triggering.

**Measurement Setup:**
- Mount a $64 \times 64$ test chip to a multi-zone Peltier thermal stage.
- **Sensor Taps:** Tap the output current of the dummy sentinel column ($I_{ref\_out}$) and simultaneously probe 4 active data columns (Corners: C0, C63, and Center: C31).

**Injected Disturbance:**
- Drive a $40^\circ C$ gradient diagonally across the chip (Corner 0 is $80^\circ C$, Corner 63 is $40^\circ C$).
- Run a high-density $R=9$ data stream through the active columns.

**Pass/Fail Criteria:**
- **Pass:** The $N_{est}$ calculated from the sentinel deviates $< 10\%$ from the true noise variance measured at the furthest active column (C63).
- **Expected Failure Signature:** If the deviation $> 10\%$, the sentinel architecture must be upgraded from a single column to a "distributed border" (sentinels on both the first and last columns, averaged).

---

## 2. Boundary Condition Test (Mixed-Mode Interface)

**Objective:** Validate that data crossing from an $R_{eff}=3$ tile into an $R_{eff}=9$ tile is not corrupted by residual analog "slop" or line parasitics.
**Hypothesis:** The $R=3$ tile emits "pure" center states (e.g., exactly State 7 voltage), but parasitic wire resistance will cause IR drop, meaning the $R=9$ tile receives a degraded voltage (e.g., State 6.5) and incorrectly bins it.

**Measurement Setup:**
- Two adjacent $64 \times 64$ tiles interconnected by physical routing buses.
- Tile A locked to $R=3$ (Survival Mode). Tile B locked to $R=9$ (High Precision).
- **Sensor Taps:** Probe the interconnect wire at the output of Tile A ($V_{out}$) and the input buffer of Tile B ($V_{in}$).

**Injected Disturbance:**
- Modulate the clock frequency to induce varying levels of inductive cross-talk on the interconnect bus.

**Pass/Fail Criteria:**
- **Pass:** The $R=9$ tile correctly bins the incoming signal $>99.9\%$ of the time, despite the IR drop.
- **Expected Failure Signature:** The $R=9$ tile consistently reads 1 state lower than intended due to IR drop. If this occurs, Tile A must apply a pre-calculated $+V_{boost}$ to all outgoing inter-tile transmissions.

---

## 3. $\beta$-Bias Saturation Test (IR Drop & Cross-Talk)

**Objective:** Ensure the $\beta$-bias DC current injected during $\phi$-mode reaches the far end of the array without saturating adjacent inactive wordlines.
**Hypothesis:** Injecting a continuous $50$ nA bias into Row 0 will successfully deepen the energy well for Column 0, but by Column 63, wire resistance will dissipate the bias, causing far-edge cells to fail. Furthermore, capacitive coupling will leak the bias into Row 1.

**Measurement Setup:**
- Single $256 \times 256$ tile (Maximum size stress test).
- Lock to $R_{eff}=3$.
- **Sensor Taps:** Micro-probe the active wordline at C0, C128, and C255. Probe the adjacent inactive wordline.

**Injected Disturbance:**
- Inject the standard $\beta$-bias current into the active row. Add $5.0$ Gaussian noise to the entire substrate.

**Pass/Fail Criteria:**
- **Pass:** The bias amplitude at C255 is $>80\%$ of the amplitude at C0. Leakage into the adjacent inactive row is $<5\%$.
- **Expected Failure Signature:** C255 fails to maintain $96\%$ logical fidelity (as predicted by simulation) because the bias has decayed. If this occurs, the maximum allowable tile size for Level 3 logic must be formally reduced (e.g., capped at $64 \times 64$).

---

## 4. Rollback Thermal Runaway Test

**Objective:** Prove that the 13x energy penalty of Level 3 Survival Mode, combined with the delayed hysteresis exit path, does not melt the chip or permanently alter $\tau_{leak}$.
**Hypothesis:** Operating in $R=3$ for an extended period generates excessive heat. Because the Slow Drift Estimator ($N_{est}$) sees this self-generated heat as "environmental noise," it creates a feedback loop: the chip stays in Level 3 because it is hot, and it stays hot because it is in Level 3.

**Measurement Setup:**
- Single $64 \times 64$ tile in a thermally isolated chamber ($25^\circ C$ ambient baseline).
- Standard Autonomous OS Control Law active.
- **Sensor Taps:** On-die thermistor, $N_{est}$ logger, and OS state logger.

**Injected Disturbance:**
- Externally inject heavy Gaussian noise (SNR 0.8) for exactly 10 seconds to force the OS into $R=3$ mode.
- Immediately cut the external noise to zero.

**Pass/Fail Criteria:**
- **Pass:** The OS successfully completes the $3 \rightarrow 5 \rightarrow 7 \rightarrow 9$ staged rollback within $200 \tau_m$ of the noise cutting out, and the die temperature stabilizes.
- **Expected Failure Signature:** The OS never exits $R=3$. The self-heating of the temporal integration masks the fact that the external noise is gone. If this occurs, the $N_{est}$ equation must be calibrated to dynamically subtract the known thermal footprint of Level 3 operation.
