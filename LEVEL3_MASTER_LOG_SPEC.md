# W-bit Level 3: Master Bench Log Specification

**DATE:** March 16, 2026
**STATUS:** Ready for Hardware Integration
**FORMAT:** Apache Parquet (Preferred) / CSV (Fallback)

This document formalizes the exact data schema, derived-metric formulas, event markers, and automatic fail-parser rules required to serialize and analyze the physical bench tests defined in the Level 3 architecture. 

To maintain clean separation of concerns, data is split into two namespaces: **Raw Logged Fields** (emitted directly by hardware/test rig) and **Derived Metrics** (computed post-hoc by the analysis pipeline).

---

## 1. Raw Logged Fields (`raw.*`)

Every logged row MUST contain the following fields. The minimum sampling frequency is 10 Hz (100ms interval). During `SHOCK_INJECTED` events, the system must micro-burst log at 10 kHz for $10,000 \tau_m$.

### 1.1 Metadata & Identity
| Column Name | Data Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `raw.timestamp_ns` | Int64 | $ns$ | Nanoseconds since epoch (UNIX time) or test start. |
| `raw.tile_id` | String | | Identifier for the physical tile (e.g., `T_0_0`, `T_0_1`). |
| `raw.event_marker` | Enum/Str | | Empty unless a specific test event occurs (See Sec 2). |

### 1.2 The Tradeoff Vector
| Column Name | Data Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `raw.fidelity_score` | Float32 | $\%$ | Correct state retention across the tile ($0.0 \rightarrow 100.0$). |
| `raw.energy_cost_est` | Float32 | $fJ/op$| Calculated dynamic + static power draw per operation. |
| `raw.boundary_bin_err`| Float32 | $\%$ | Error rate on the $R=3 \rightarrow R=9$ physical interconnect. |

### 1.3 Control & State Logic
| Column Name | Data Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `raw.os_mode_state` | Int8 | | Current active $R_{eff}$ regime $\{3,5,7,9\}$. |
| `raw.rollback_phase` | String | | Current state: `steady`, `downgrade`, `hold`, `step_3_5`, `step_5_7`, `step_7_9`, `recenter`. |

### 1.4 Estimator & Environment Variables
| Column Name | Data Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `raw.ambient_temp_c` | Float32 | $^\circ C$ | Chamber / External ambient temperature baseline. |
| `raw.temp_edge_c` | Float32 | $^\circ C$ | Surface temperature at the driver edge ($C_0$). |
| `raw.temp_center_c` | Float32 | $^\circ C$ | Surface temperature at the tile center ($C_{31}$). |
| `raw.temp_far_c` | Float32 | $^\circ C$ | Surface temperature at the furthest boundary ($C_{63}$). |
| `raw.known_external_heat`| Float32 | $^\circ C$ | Intentionally injected heat delta from the bench rig (e.g., Peltier offset). |
| `raw.n_est_slow` | Float32 | $mV_{var}$| Output of the slow EMA noise sentinel. |
| `raw.true_noise_c63` | Float32 | $mV_{var}$| Bench-rig ground truth measurement of noise at the far corner. |
| `raw.shock_flag` | Boolean | | State of the transient shock high-pass detector. |
| `raw.v_offset_drift` | Float32 | $mV$ | Measured deviation of the primary comparator threshold. |

### 1.5 Substrate Parasitics
| Column Name | Data Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `raw.beta_far_ratio` | Float32 | Ratio | $\beta$-bias amplitude at $C_{63} / C_0$. |
| `raw.adj_leak_ratio` | Float32 | Ratio | Current leaked to inactive row $/$ intended bias. |

---

## 2. Event Markers

The `raw.event_marker` column is left null during standard execution. The external test rig or internal OS must stamp the following exact strings to align postmortem analysis:

- `DISTURBANCE_ON`: External noise, heat, or offset is applied.
- `DISTURBANCE_OFF`: External interference is removed.
- `SHOCK_INJECTED`: EMP or high-voltage pulse fired.
- `SHOCK_CLEAR`: Fast transient shock has dissipated.
- `ROLLBACK_START`: OS leaves the `hold` phase and begins `step_3_5`.
- `ROLLBACK_COMPLETE`: OS enters `steady` $R=9$.
- `RECENTER_WRITE_START`: Final clean write before releasing control.
- `RECENTER_WRITE_COMPLETE`: Final clean write has finished.

---

## 3. Derived Metrics (`derived.*`)

These metrics are calculated via the analysis pipeline (e.g., Pandas/Polars) rather than computed on-die.

**Estimator Error ($\Delta \%$)**
*Identifies the "Silent Poison" condition.*
$$ \text{derived.n\_est\_error} = \frac{\text{raw.n\_est\_slow} - \text{raw.true\_noise\_c63}}{\text{raw.true\_noise\_c63}} \times 100 $$

**Thermal Self-Load ($^\circ C$)**
*Identifies Level 3 self-heating runaway.*
$$ \text{derived.thermal\_self\_load} = \text{raw.temp\_center\_c} - \text{raw.ambient\_temp\_c} - \text{raw.known\_external\_heat} $$

**Recovery Latency ($ms$)**
*Measures OS rollback agility.*
$$ \text{derived.recovery\_latency} = timestamp(\text{ROLLBACK\_COMPLETE}) - timestamp(\text{DISTURBANCE\_OFF}) $$

**Recovery Validity**
*Ensures the latency metric is not masking a corrupt state.*
$$ \text{derived.recovery\_valid} = \text{raw.fidelity\_score} \ge 99.0 \text{ at } ROLLBACK\_COMPLETE $$

---

## 4. Automatic Fail-Parser Rules

The postmortem analysis script runs these exact boolean queries across the dataset. *Note: These pandas rules assume `df` has a proper DateTime or TimeDelta index created from `raw.timestamp_ns` and is sorted chronologically.*

### Rule 1: Thermal Runaway
```python
fail_runaway = (df['raw.os_mode_state'] == 3) & (df['derived.thermal_self_load'] > 15.0)
```
### Rule 2: Silent Poison
```python
spatial_delta = (df['raw.temp_edge_c'] - df['raw.temp_far_c']).abs()
fail_poison = (spatial_delta > 10.0) & (df['derived.n_est_error'] < -15.0) & (df['raw.os_mode_state'] == 9) & (df['raw.fidelity_score'] < 95.0)
```
### Rule 3: OS Thrashing
```python
# Fails if OS state changes more than 3 times within a 1-second rolling window
fail_mode_thrashing = df['raw.os_mode_state'].diff().ne(0).rolling('1s').sum() > 3
# Fails if rollback phase flaps more than 5 times in 1 second (internal instability)
fail_phase_thrashing = (df['raw.rollback_phase'] != df['raw.rollback_phase'].shift(1)).rolling('1s').sum() > 5
fail_thrashing = fail_mode_thrashing | fail_phase_thrashing
```
### Rule 4: False-Shock Paralysis
```python
# Fails if shock_flag is True for >100ms WITHOUT a SHOCK_INJECTED event occurring in that window
has_real_shock = df['raw.event_marker'].eq('SHOCK_INJECTED').rolling('100ms').max().fillna(False)
is_paralyzed = (df['raw.shock_flag'] == True).rolling('100ms').min() == 1
fail_paralysis = is_paralyzed & ~has_real_shock
```
### Rule 5: Energy Budget Violation
```python
# Fails if energy draw exceeds budget for 5 continuous seconds
fail_energy = (df['raw.energy_cost_est'] > 1000.0).rolling('5s').min() == 1
```