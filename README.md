# W-bit

**W-bit** (Weight-Defined Non-Binary Primitive) is an alternative, non-binary logic
substrate for ML-style inference and approximate computation. Instead of Boolean gates
and an ALU, computation is expressed as **energy relaxation over a radix-R weighted
network**: each cell holds one of `R` discrete states, and the behavior of the network is
defined entirely by its weights.

W-bit is **not** a drop-in binary CPU replacement (see [Non-goals](#non-goals)). It is a
research substrate: this repository validates the primitive in simulation and provides an
RTL implementation of the supporting datapath.

## What W-bit is

Each cell `i` is in one of `R` states (or, in the continuous formulation, a distribution
over `R` states). Two sets of weights define all behavior:

- a **local bias** vector `theta_i` (length `R`) — the cell's intrinsic preference, and
- an **R×R pairwise compatibility matrix** `Theta_ij` — how the state of cell `j` biases
  cell `i`.

The network relaxes toward a low-energy configuration of the energy function

```
E(S) = - Σ_i  theta_i · s_i  - Σ_ij  s_iᵀ Theta_ij s_j
```

which has the form of a **Potts model / pairwise Markov random field**. Equivalently — and
this is what the code does — it *maximizes* the compatibility `H(S) = −E(S)` (the positive
sum of the same terms): `network.py`'s `get_energy()` returns `H`, and `step()` moves each
cell to its highest-scoring state. Inference is this relaxation, and the repo implements it
two ways:

- **Deterministic** — an argmax/ICM sweep (`src/wbit/network.py`, `mode='deterministic'`),
  which iterates each cell to its best-scoring state until stable.
- **Stochastic** — a softmax/Gibbs update at temperature `T`
  (`network.py` `mode='stochastic'`), and a continuous **mean-field** relaxation with
  inertia and injectable analog noise (`src/wbit/analog_network.py`).

"Logic" is therefore programmed by *setting weights*, not by wiring gates. The examples in
[`examples/`](examples/) build a hand-coded router and search for a nonlinear gate (a
`MAX(A, B)` function) purely by choosing `theta` and `Theta`.

## The Aeternum radix link

W-bit shares a single radix parameter with the Aeternum CPU project. In adaptive mode the
radix is constrained to `R = 2n + 1`, so the valid radices

```
R ∈ {3, 5, 9, 17, 33}   ↔   Aeternum n ∈ {1, 2, 4, 8, 16}
```

are enforced in `AnalogWbitNetwork` (`src/wbit/analog_network.py`), keeping the W-bit state
alphabet aligned with Aeternum's signed value range `[-n, +n]`. This alignment lives in the
Python model — the RTL itself carries no radix parameter (its cells are fixed 8-bit); the
RTL side of the link is the `wbit_if` interface that exposes the fabric to the Aeternum core
(`rtl/wbit_if.sv`, `rtl/wbit_accelerator.sv`).

## What's in the repo

### Core model — `src/wbit/`
- **`network.py`** — `WbitNetwork`: the reference pure-Python R-state model (discrete
  state, local bias, sparse `R×R` compatibility matrices, deterministic/stochastic
  relaxation).
- **`analog_network.py`** — `AnalogWbitNetwork`: the continuous/analog formulation. State
  is a distribution over `R` states, updated by a mean-field softmax step with inertia and
  Gaussian noise; includes the Aeternum radix constraint and a Relaxation Cost Proxy (RCP)
  metric.
- **`level3_wbit_network.py`** — `Level3WbitNetwork`: adds a **runtime adaptive-capacity
  scheduler**. A cell has a maximum radix `R_max`; `schedule_capacity()` drops the
  effective radix `R_eff` under high noise / low task complexity (down to `R_eff = 3` for
  robustness) and raises it back toward `R_max` when conditions are clean.
- **`level3_temporal_wbit_network.py`** — `Level3TemporalWbitNetwork`: a temporal variant
  of the same idea, integrating a per-state voltage over time (leak, threshold, hysteresis)
  instead of an instantaneous softmax.
- **`quantization_lab.py`** — quantizes real weight matrices into `R` evenly-spaced levels,
  injects weight noise, and scores reconstruction MSE (the engine behind
  `exp_real_layer_quant.py`). **`step_utils.py`** — relaxation-step summaries.

### RTL — `rtl/`
A SystemVerilog implementation of the matrix-vector-multiply (MVM) datapath the model
relies on. This is the compute substrate, not a full hardware realization of the R-state
relaxation:
- **`wbit_tile.sv`** — a weight-stationary, pipelined MAC array (256×256, 8-bit signed
  weight cells), with an online Hebbian weight-update port and optional LFSR read-noise.
- **`wbit_update_engine.sv`** — combinational row-parallel Hebbian update
  (`W_new = W + (x·y >> shift)`, saturated to 8-bit).
- **`wbit_lfsr.sv`** — 16-bit LFSR used as a read-noise source.
- **`wbit_accelerator.sv`** — the tile wrapped with a forward/learn FSM; **`wbit_noc.sv`** —
  a standalone ring network-on-chip node (a scaling sketch; not wired into the default build).
- **`wbit_if.sv`** — the Aeternum-core ↔ W-bit-fabric interface.
- **`wbit_tb.sv`**, **`sim_main.cpp`** (Verilator, self-checking), **`sim_verilator.sh`**
  (Verilator build), **`synth_wbit.ys`** (Yosys synthesis, reports cell `stat` estimates).
  The build/sim/synth flow covers the tile, update engine, accelerator, and interface
  (`wbit_top`); the NoC is not included.

A second, evolved RTL set lives under `W-bit-Storage/rtl/` (see [Companion
components](#companion-components-w-bit-storage)), including `wbit_copro_wrapper.sv`, an
Aeternum-coprocessor → W-bit-accelerator bridge with a documented command/bus protocol.

### Experiments — `experiments/`
43 scripts exercising the substrate. The canonical Phase 1 suite (driven by
`run_phase1.py`) is:
- **Experiment A — Router Sweep** (`exp_a_router_sweep.py`): gradient/"liquid" routing on a
  grid across noise, radix, obstacle density, and layouts.
- **Experiment B — Noise Breakdown** (`exp_b_noise_breakdown.py`,
  `exp_b_weight_noise_grid.py`): robustness of gate logic under increasing analog noise.
- **Experiment C — Learning Search** (`exp_c_learning_search.py`): evolutionary/population
  search for nonlinear (XOR) learning capacity (needs hidden cells and a population > 1).

The broader set includes: Experiment D low-rank/**WZMA reconstruction**
(`exp_d_wzma_reconstruction.py`, ~0.90 cosine fidelity); Level-3 **adaptive-logic** proofs
and ablations (`proof_of_adaptive_logic.py`, `ablation_*`, `exhaustive_ablation.py`,
`test_temporal_wbit.py`); noise/robustness baselines (`base_noise_test*`,
`test_binary_noise.py`, `verify_champion_noise*`, `adaptive_policy_search.py`);
KV/bottleneck/sparsity studies (`exp_e`–`exp_h`); hyperparameter tuning
(`tune_wbit_hyperparams.py`) and layer quantization (`exp_real_layer_quant.py`); an energy
profiler (`energy_profiler.py`); and endurance/throughput stress harnesses, including
PyTorch- and ROCm/GPU-accelerated variants against CPU baselines
(`million_cycle_stress_test.py`, `billion_cycle_stress_test.py`, `rocm_billion_cycle.py`,
`cpu_billion_*.py`). Note the billion-cycle and reconstruction scripts require PyTorch,
which is beyond the minimal `requirements.txt` (matplotlib + pytest).

### Analysis, tests, tools
- **`analysis/`** — aggregation and plotting: Phase 1/2 report aggregation
  (`aggregate_phase1_report.py`, `aggregate_phase2_report.py`), per-experiment plots
  (`plot_exp_*`), phase diagrams, and a Phase 2 delta/Pareto/frontier analytics layer
  (`phase2_{delta,pareto,frontier,steps_audit}.py` + plotters).
- **`tests/`** — a `pytest` suite: path hygiene, no-pandas import guard, plasticity and
  WZMA-reference checks, and Phase 1/2 smoke tests (`smoke_phase1_8.py`,
  `smoke_phase2_scaffold.py`).
- **`tools/`** — Level-3 run reporting/parsing over bench logs (`level3_fail_parser.py`,
  `level3_plot_report.py`, `level3_run_metadata.yaml`).

### Docs & paper
- [`docs/WBIT_REPORT.md`](docs/WBIT_REPORT.md) — architecture report and simulation
  findings.
- [`docs/WBIT_EXPERIMENTS.md`](docs/WBIT_EXPERIMENTS.md) — reproducible experiment
  protocols and metrics.
- [`paper/WBIT_WHITE_PAPER.md`](paper/WBIT_WHITE_PAPER.md) — the white paper.

### Companion components (`W-bit-Storage/`)

Alongside the core substrate, the repo ships a **WZMA reference model**
(`src/wzma_reference/`) — a small factorized-weight transformer with online Hebbian
plasticity, plus a training/eval pipeline and a 3-way on-chip context-memory benchmark
(ReRAM vs. holographic vs. sparse-distributed, `memory_bench.py`). The
`exp_d_wzma_reconstruction.py` experiment above reconstructs this model's low-rank weight
patches on the W-bit substrate.

The **`W-bit-Storage/`** subtree is a related, more exploratory workspace with its own
`src/`, `rtl/`, and `tests/`. It contains the **Aeternum coprocessor bridge RTL**
(`rtl/wbit_copro_wrapper.sv`, a documented command/bus protocol between the Aeternum core
and the W-bit accelerator), an evolved copy of the WZMA reference, and a Neural-File-System
simulator sketch (`src/neural_fs_sim.py`). These components require PyTorch and are not
covered by the Phase 1/2 validation flow; treat them as adjacent work rather than part of
the validated core. Trained model checkpoints are not committed — the pipelines regenerate
them.

## Current status vs. stated aim

**Validated in this repo (simulation + RTL):**
- The primitive works as a substrate — a committed run of each is under `results/` (and is
  regenerated by re-running the experiments):
  - **Routing** — the Exp A sweep solves grid routing across obstacle densities and layouts.
  - **Noise robustness** — in the Exp B sweep the `R=3` gate logic keeps correct states
    across the full swept range (σ up to 2.0) at ~99%+ success.
  - **Nonlinear (XOR) learning** — Exp C solves XOR *only with hidden cells* (`H=0` fails;
    with a real search population, `H≥1` converges to MSE < 0.02). Note the *search* needs a
    population > 1: the bare default `--population 1` will not converge — `run_phase1.py` now
    uses a real population, and the quick-start command below shows one.
  - **Low-rank reconstruction** — Exp D reconstructs low-rank WZMA weight patches at ~0.90
    mean cosine fidelity (`R=5`, σ=0.2).
- A **runtime adaptive-capacity scheduler** (`R_eff` vs. noise) is implemented and ablated
  (`level3_wbit_network.py`, `ablation_*`, `exhaustive_ablation.py`).
- An **RTL datapath** (tile / update engine / accelerator / Aeternum interface) that
  compiles, simulates under Verilator (self-checking `sim_main.cpp`), and synthesizes to
  cell-count estimates under Yosys.

**Stated aim / vision (not demonstrated here):**
- Mapping W-bit onto a **physical memristive substrate** (e.g. HfZrO / "Morphium-E"
  crossbars) and any resulting **energy advantage over CMOS** — the docs explicitly frame
  this as a hypothesis pending device calibration and physical prototyping.
- **Hosting a large model directly in analog hardware** and **self-modifying conductances**
  in real time (the "Level 3" direction). These are goals described in the white paper, not
  results in this repository.

Please treat anything in the second list as direction, not achievement.

## Quick start

Run the full Phase 1 validation suite with default parameters:

```bash
python3 run_phase1.py
```

Results are written under `results/expA/`, `results/expB/`, and `results/expC/`. For a fast
check:

```bash
python3 run_phase1.py --smoke
```

Phase 2 (binary/adaptive baseline comparison) scaffold:

```bash
python3 run_phase2.py --smoke --run_expB_grid
```

Run the hand-coded examples directly:

```bash
python3 examples/router_demo.py     # weight-defined 1-to-2 router (discrete)
python3 examples/train_logic.py     # random search for a MAX(A,B) gate (R>2)
python3 examples/grid_router.py     # 10x10 "liquid" routing fabric around obstacles
python3 examples/analog_router.py   # continuous, memristor-like settling router
python3 examples/train_analog.py    # train the analog network on a logic task
```

### Running individual experiments

```bash
python3 experiments/exp_a_router_sweep.py --trials 100 --sigma 0.1 --grid 10
python3 experiments/exp_b_noise_breakdown.py --trials 100 --R 3
# Exp C needs a real search population to solve XOR (the default --population 1 will not):
python3 experiments/exp_c_learning_search.py --trials 5 --R 2 --H 0 1 2 4 \
    --population 30 --elite_k 6 --restarts 5
```

- Exp A `--obstacle_density`: densities to sweep (default `0.1 0.2 0.3`).
- Exp B `--sigma`: specific noise level; if omitted, sweeps 0.0 → 2.0 (step 0.1).
- Exp C `--H`: hidden-cell counts to sweep (default `0 1 2 4`); `--population`/`--restarts`
  control the search (defaults are 1 — increase them, or XOR will not converge).

### Metric

Experiments log the **Relaxation Cost Proxy (RCP)**, an internal heuristic for relative
algorithmic cost (distinct from physical energy):

```
RCP = (N_cells * N_steps * R^2) / I_out
```

Logs are saved as `results.csv` in each result directory.

### Optional virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### RTL (optional)

```bash
cd rtl
./sim_verilator.sh      # build + run the Verilator testbench (needs Verilator)
yosys synth_wbit.ys     # synthesis stat estimates (needs Yosys)
```

## Paths & outputs

- Repo root: `W-bit/`
- Outputs are written under `results/`:
    - Exp A → `results/expA`
    - Exp B (noise breakdown) → `results/expB`
    - Exp B (grid, optional) → `results/expB_grid`
    - Exp C → `results/expC`
    - Aggregated report → `results/phase1_report.csv`

`run_phase1.py` uses these paths by default; pass `--smoke` for fast checks and
`--run_expB_grid` to include the grid sweep. Phase 2 outputs go under `results/phase2/`
(`wbit/`, `binary/`, `adaptive/`), with aggregation in
`analysis/aggregate_phase2_report.py` and plots in `analysis/plot_phase2_comparison.py`.

## Non-goals

W-bit is a specialized substrate, not a general-purpose processor. It is **not** intended
for:

- **Exact arithmetic** or bit-exact serial computation,
- **Serial algorithmic processing** (OS kernels, compilers, traditional code execution),
- **Dense storage** (standard DRAM/SRAM is far denser), or
- being a **drop-in binary CPU replacement**.

Its target is ML-style inference/learning and approximate, noise-tolerant computation where
weight-defined, parallel energy relaxation is a natural fit.
