# w-dit Next Steps (Prioritized)

## Phase 1: Validation (Current Codebase)
- [ ] **Run Experiment B Sweep:** Modify `analog_router.py` to loop noise $\sigma$ from 0.0 to 1.0 and plot accuracy.
- [ ] **Run Experiment C Structure Search:** Modify `train_analog.py` to loop Hidden Neurons $H=[0,1,2,3]$ and report convergence rates.
- [ ] **Implement Metric Logging:** Add `calculate_ROPB()` function to `AnalogWDitNetwork` and print it in all demos.

## Phase 2: Scaling (New Features)
- [ ] **Vectorized Engine:** Rewrite `analog_network.py` using `numpy` (if available in future environment) for 100x speedup to support $N > 1000$ grids.
- [ ] **Dynamic Routing:** Create `packet_simulator.py`. Instead of just calculating the field, actually move a "packet" (state) across the grid in real-time as the field updates.
- [ ] **Memristor Noise Model:** Upgrade noise model from simple Gaussian to "Stuck-At" faults (simulating broken hardware cells).

## Phase 3: Hardware Design
- [ ] **Draft Verilog:** Write a simplified Verilog behavioral model of a single w-dit cell to estimate silicon area.
- [ ] **Power Estimation:** Estimate Joules/Op based on analog memristor literature (e.g., 1pJ per weight update) vs the simulated steps.
