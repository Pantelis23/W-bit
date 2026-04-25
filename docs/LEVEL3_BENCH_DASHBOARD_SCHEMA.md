# W-bit Level 3: Bench Dashboard & Master Log Schema

**DATE:** March 16, 2026
**PURPOSE:** Standardize data logging and automatic fail criteria across all bench tests.

To ensure the tradeoff controller (Precision vs. Robustness vs. Heat vs. Energy) is functioning correctly, every bench test MUST log the following standardized data frame at a minimum frequency of 10 Hz (with 10 kHz micro-burst logging during shock events).

---

## 1. Master Log Vector (Logged Continuously)

| Metric | Source | Unit | Description |
| :--- | :--- | :--- | :--- |
| `os_mode_state` | Control Bus | $\{3,5,7,9\}$ | Current active $R_{eff}$ regime. |
| `rollback_phase` | OS Scheduler | Enum | Current transition state: `{steady, downgrade, hold, step_3_5, step_5_7, step_7_9, recenter}`. |
| `fidelity_score` | Data Bus vs. Ground Truth | $\%$ | Percentage of cells holding correct logical state. |
| `boundary_bin_err`| Tile Interconnect | $\%$ | Rate of binning errors across mixed-mode ($R=3 \rightarrow R=9$) boundaries. |
| `energy_cost_est` | Power Rails | $fJ / op$ | Instantaneous dynamic + static power draw converted to per-operation cost. |
| `temp_edge_C` | Thermal Tap (C0) | $^\circ C$ | Surface temperature at the driver edge. |
| `temp_center_C` | Thermal Tap (C31) | $^\circ C$ | Surface temperature at the tile center. |
| `temp_far_C` | Thermal Tap (C63) | $^\circ C$ | Surface temperature at the furthest boundary. |
| `thermal_self_load`| Energy/Temp Model | $^\circ C$ | Estimated heat contributed solely by Level 3 operation (self-heating). |
| `N_est_slow` | Sentinel Dummy Column | $mV_{var}$ | The current output of the slow EMA noise estimator. |
| `N_est_error` | Sentinel vs. C63 Probe| $\Delta \%$ | Deviation between sentinel estimated noise and true physical noise at the far edge. |
| `shock_flag` | High-Pass Filter | Boolean | State of the transient shock detector. |
| `v_offset_drift` | ADC Calibration | $mV$ | Deviation of the primary comparator threshold from factory baseline. |
| `beta_far_ratio` | $\beta$-Bias Monitor | Ratio | Amplitude of $\beta$-bias at far edge ($C_{far}$) vs driver edge ($C_0$). Tracks IR drop. |
| `adj_leak_ratio` | $\beta$-Bias Monitor | Ratio | Current leakage into adjacent inactive wordlines. Tracks cross-talk saturation. |

---

## 2. Automatic Red Flags (Immediate Test Failure)

If any of the following conditions are met during a run, the test is automatically flagged as a **CATASTROPHIC FAILURE**, regardless of the logical fidelity score.

1. **The Thermal Runaway Flag:** 
   - `temp_center_C` rises $> +15^\circ C$ above ambient solely due to the chip operating in Level 3 Survival Mode without external heating.
2. **The "Silent Poison" Flag:**
   - The spatial temperature delta (`temp_edge_C` minus `temp_far_C`) exceeds $10^\circ C$ AND the $N_{est\_slow}$ fails to trigger a mode downgrade before `fidelity_score` drops below $95\%$.
3. **The OS Thrashing Flag:**
   - `os_mode_state` changes more than 3 times within a 1-second window (violates hysteresis laws).
4. **The False-Shock Paralysis Flag:**
   - `shock_flag` remains TRUE for $> 100 \tau_m$, locking the bitlines out of computation when no actual cosmic ray/burst event occurred.
5. **The Energy Budget Violation:**
   - `energy_cost_est` exceeds $1,000$ $fJ/op$ for $> 5$ continuous seconds.

---

## 3. Supplementary Bench Tests (Added per Review)

### 5. Comparator Offset Drift Test
**Objective:** Validate that the OS and ADC margins can tolerate inevitable manufacturing and thermal drift of the comparator thresholds.
**Measurement Setup:** 
- Inject a controlled DC offset ($0 \rightarrow -500 mV$) directly into the comparator reference voltage ($V_{ref}$) during active execution.
**Pass/Fail:**
- **Pass:** The $R=3$ $\phi$-mode correctly absorbs up to $300mV$ of offset without dropping `fidelity_score` below $99\%$.
- **Expected Failure Signature:** A clean, sudden drop in fidelity on one specific bin edge. If it occurs at $< 300mV$, the physical WZMA rank (basin width) must be increased.

### 6. Shock Path False-Positive / Accuracy Test
**Objective:** Measure the accuracy and recovery latency of the Fast Transient channel independently of the slow estimator.
**Measurement Setup:**
- Fire an EMP gun or high-voltage pulse generator near the unshielded test board.
**Pass/Fail:**
- **Pass:** The `shock_flag` asserts within $1 ns$, floating bitlines before data corruption occurs. The system recovers and drops the flag exactly $5 \tau_{leak}$ after the pulse ends. Zero false positives during clean operation.

### 7. Multi-Duration Stress Cycling
**Objective:** Prove that repeated entry/exit of Survival Mode does not cause accumulating estimator bias or $\tau$-shift creep.
**Measurement Setup:**
- Cycle the external noise source: 10 seconds ON, 1 minute OFF, 1 hour ON, 5 minutes OFF, repeated for 48 hours.
**Pass/Fail:**
- **Pass:** The rollback hysteresis timing ($100 \tau_m$) remains perfectly consistent on the 100th cycle as it was on the 1st cycle. The `v_offset_drift` metric does not show a permanent accumulating shift.