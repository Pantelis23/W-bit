# w-bit Phase 1 Validation

This directory contains the reproducible simulation suite for the w-bit architecture (older docs may say "w-bit"—same project).

## Project Identity

**w-bit** is an alternative non-binary, weighted, noisy logic substrate intended for ML-style inference/learning and approximate control, not a drop-in binary CPU replacement. Phase 1 is a controlled validation suite (Exp A/B/C) with reproducibility safeguards (map invariance, path hygiene, fixed outputs under `results/`).

## Overview

Three experiments are implemented to validate the core claims:
*   **Experiment A (Router Sweep):** Statistical verification of routing efficiency on a 10x10 grid.
*   **Experiment B (Noise Breakdown):** Robustness testing of logic gates under increasing analog noise.
*   **Experiment C (Learning Search):** Structural search to confirm non-linear learning capability (XOR).

## Quick Start

To run the full Phase 1 validation suite with default parameters:

```bash
python3 run_phase1.py
```

Results will be generated in `results/expA/`, `results/expB/`, and `results/expC/`.

For a quick validation run, use:

```bash
python3 run_phase1.py --smoke
```

Phase 2 scaffold (binary baseline comparison) is available via:

```bash
python3 run_phase2.py --smoke --run_expB_grid
```

## Running Individual Experiments

You can run each experiment manually with custom parameters.

### Experiment A: Router Sweep

```bash
python3 experiments/exp_a_router_sweep.py --trials 100 --sigma 0.1 --grid 10
```
*   `--obstacle_density`: List of densities to sweep (e.g., `0.1 0.2 0.3`). Default is `[0.1, 0.2, 0.3]`.

### Experiment B: Noise Breakdown

```bash
python3 experiments/exp_b_noise_breakdown.py --trials 100 --R 3
```
*   `--sigma`: Specific noise level. If omitted, runs a sweep from 0.0 to 1.0 (step 0.1).

### Experiment C: Learning Search

```bash
python3 experiments/exp_c_learning_search.py --trials 10 --R 2
```
*   `--H`: List of hidden neuron counts to sweep (e.g., `0 1 2 4`). Default is `[0, 1, 2, 4]`.

## Metrics

All experiments log the **Relaxation Cost Proxy (RCP)**, defined as:
`RCP = (N_cells * N_steps * R^2) / I_out`

Logs are saved as `results.csv` in the respective result directories.

## Paths & Outputs

* Repo root: `wdit_project/`
* Outputs are under `results/`
    * Exp A → `results/expA`
    * Exp B (noise breakdown) → `results/expB`
    * Exp B (grid, optional) → `results/expB_grid`
    * Exp C → `results/expC`
    * Aggregated report → `results/phase1_report.csv`

`run_phase1.py` uses these paths by default; pass `--smoke` for fast checks and `--run_expB_grid` to include the grid sweep.

## Phase 2 Plan (Baseline Comparisons)

Phase 2 now includes a binary baseline scaffold to compare against w-bit:
* Runner: `run_phase2.py` (supports `--smoke`, `--run_expB_grid`)
* Outputs: `results/phase2/wbit/...` and `results/phase2/binary/...`
* Aggregation: `analysis/aggregate_phase2_report.py`
* Goal: side-by-side summaries/plots showing regimes where w-bit is more noise-robust or achieves lower RCP at comparable success. Binary mode now runs with enforced binary states; plotting via `analysis/plot_phase2_comparison.py`.

## Local Venv (optional)

Create and activate a local virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```
