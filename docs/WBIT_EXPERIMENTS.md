# w-bit Experiment Protocols v3.0

This document defines the reproducible study specifications for verifying the w-bit architecture (historical filenames use "w-bit"). All experiments utilize the `AnalogWDitNetwork` engine to ensure physical realism.

## 1. Unifying Metric
**Metric:** Relaxation Cost Proxy (RCP)
**Definition:** A heuristic cost function derived from simulation steps and network size, used to compare relative efficiency of w-bit configurations.
**Goal:** Minimize RCP.

## 2. Experiment A: The Liquid Router

**Objective:** Quantify routing latency and obstacle avoidance success rate in a 2D mesh.

### 2.1 Setup
*   **Grid Size:** $10 \times 10$ ($N=100$)
*   **Radix:** $R=5$ (Idle, N, E, S, W)
*   **Target:** Fixed at $(9, 9)$.
*   **Obstacles:** Randomly placed (Density $\rho_{obs} \in \{0.1, 0.2, 0.3\}$).

### 2.2 Parameters
*   **Noise Model:** Gaussian $\mathcal{N}(0, 0.1)$
*   **Temperature:** $T=0.2$
*   **Time Constant:** $\Delta t = 0.5$
*   **Max Steps:** 50

### 2.3 Metrics
*   **Success Rate:** % of trials where a valid path is formed from $(0,0) \to (9,9)$.
*   **Convergence Latency ($\tau_{99}$):** Steps required for path cells to reach >0.99 probability.
*   **Baseline:** Standard A* algorithm (Software). Comparison is heuristic: w-bit "settling steps" vs A* "visited nodes".

### 2.4 Expected Failure Modes
*   **Local Minima:** "Dead ends" (Box canyons) where the gradient field cannot guide the packet out without lookahead.

## 3. Experiment B: Noise Robustness (Gating)

**Objective:** Determine the breakdown point of w-bit logic under signal noise.

### 3.1 Setup
*   **Topology:** 1 Control Cell (C), 1 Data Cell (D), 2 Output Cells (O1, O2).
*   **Function:** 1-to-2 Demux.
*   **Radix:** $R=3$

### 3.2 Protocol (Sweep)
Sweep Noise Level $\sigma$ from $0.0$ to $2.0$ in increments of $0.1$.
*   **Trials:** 100 per $\sigma$.
*   **Criterion:** "Success" if output matches truth table with probability $>0.8$.

### 3.3 Metrics
*   **Noise Rejection Ratio (NRR):** $\frac{\sigma_{breakdown}}{V_{signal}}$, where $V_{signal} \approx 1.0$ (normalized probability range).
*   **Expected Result:** Logic likely robust up to $\sigma \approx 0.5$ (Signal-to-Noise Ratio $\approx 2:1$).

## 4. Experiment C: Non-Linear Learning (XOR)

**Objective:** Establish the minimum structural complexity required for non-linear logic.

### 4.1 Setup
*   **Inputs:** 2 (A, B)
*   **Output:** 1 (Y)
*   **Hidden Layers:** Variable $H \in \{0, 1, 2, 4\}$
*   **Radix:** $R=2$
    *   *Note:* XOR is included as a minimal non-linear separability sanity check; $R=2$ is a special case used to validate compositional non-linearity before scaling to $R>2$ tasks.

### 4.2 Training Protocol
*   **Algorithm:** Genetic Algorithm / Random Mutation Hill Climbing.
*   **Population:** 1 (Greedy).
*   **Mutation Rate:** Adaptive ($1.0 \to 0.1$).
*   **Termination:** MSE $< 0.02$ or Steps $> 2000$.

### 4.3 Metrics
*   **Learning Efficiency:** Epochs to convergence.
*   **RCP:** Calculated at inference time.
*   **Hypothesis:** $H=0$ and $H=1$ will fail to converge reliably. $H \ge 2$ is required.

## 5. Execution Plan
To validate these experiments, updated python scripts should accept command-line arguments for $\sigma$, $N$, and $H$.
