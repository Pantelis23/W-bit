# w-bit Next Steps (Prioritized) v3.0

## Phase 1: Validation (Priority: Statistical Confirmation)
1.  **Experiment A (Router Sweep):** Run 100 trials of the 10x10 Grid Router with random obstacles to determine statistically significant success rates and average latency.
2.  **Experiment B (Noise Breakdown):** Modify `analog_router.py` to loop noise $\sigma$ from 0.0 to 1.0 and plot the "cliff" where logic fails.
3.  **Experiment C (Learning Search):** Run `train_analog.py` across $H=[0,1,2,3]$ to confirm the "2-hidden-neuron" threshold for XOR convergence.

## Phase 2: Binary Baseline + Scaling (Comparative)
1.  **Binary Baseline:** Run all Phase 1 experiments twice (`--mode wbit` vs `--mode binary`) via `run_phase2.py`, storing outputs under `results/phase2/{wbit,binary}/...`.
2.  **Aggregation & Plots:** Use `analysis/aggregate_phase2_report.py` and `analysis/plot_phase2_comparison.py` to produce summaries/overlays showing noise-robustness regimes where w-bit outperforms or matches binary at lower RCP.
3.  **Vectorized Engine (optional):** If numpy is permitted, prototype a vectorized path to speed larger grids.
4.  **Dynamic Routing (optional):** Implement `packet_simulator.py` to visualize real-time packet movement against the evolving potential field.

## Phase 3: Hardware Modeling
1.  **Device-Level Noise:** Upgrade noise model from simple Gaussian to "Stuck-At" faults (simulating broken hardware cells) and non-linear weight saturation.
2.  **Physical Cost Estimation:** Estimate area/power using literature values for generic memristor crossbars, mapped to the simulation's step counts.
