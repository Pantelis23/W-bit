# w-dit Experiment Protocols v2.0

This document defines the reproducible study specifications for verifying the w-dit architecture. All experiments utilize the `AnalogWDitNetwork` engine to ensure physical realism.

## 1. Unifying Metric
**Metric:** Relaxation Operations Per Bit (ROPB)
**Definition:** The total computational operations (Matrix-Vector Multiplies) required to resolve 1 bit of information at the output with >99% confidence.
**Goal:** Minimize ROPB.

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
*   **Baseline:** Standard A* algorithm (Software). Compare w-dit "settling steps" vs A* "visited nodes".

### 2.4 Expected Failure Modes
*   **Local Minima:** "Dead ends" (Box canyons) where the gradient field cannot guide the packet out without lookahead.

## 3. Experiment B: Noise Robustness (Gating)

**Objective:** Determine the breakdown point of w-dit logic under signal noise.

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
*   **Expected Result:** Logic should hold up to $\sigma \approx 0.5$ (Signal-to-Noise Ratio $\approx 2:1$).

## 4. Experiment C: Non-Linear Learning (XOR)

**Objective:** Establish the minimum structural complexity required for non-linear logic.

### 4.1 Setup
*   **Inputs:** 2 (A, B)
*   **Output:** 1 (Y)
*   **Hidden Layers:** Variable $H \in \{0, 1, 2, 4\}$
*   **Radix:** $R=2$

### 4.2 Training Protocol
*   **Algorithm:** Genetic Algorithm / Random Mutation Hill Climbing.
*   **Population:** 1 (Greedy).
*   **Mutation Rate:** Adaptive ($1.0 \to 0.1$).
*   **Termination:** MSE $< 0.02$ or Steps $> 2000$.

### 4.3 Metrics
*   **Learning Efficiency:** Epochs to convergence.
*   **ROPB:** Calculated at inference time.
*   **Hypothesis:** $H=0$ and $H=1$ will fail to converge reliably. $H \ge 2$ is required.

## 5. Execution Plan
To validate these experiments, update the existing python scripts in `examples/` to accept command-line arguments for $\sigma$, $N$, and $H$.
