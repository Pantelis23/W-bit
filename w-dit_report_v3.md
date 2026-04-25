# w-bit Architecture Report v3.0

**Status:** Draft (Internal)  
**Date:** December 2025  
**Subject:** Weight-Defined Non-Binary Base Primitive (w-bit; historical "w-dit" naming)  

## 1. Executive Summary

The **w-bit** is a non-binary computing primitive that substitutes boolean algebra with **R-ary energy minimization**. Unlike von Neumann architectures that rely on high-speed serial switching of binary states, w-bit relies on the parallel relaxation of a multi-valued compatibility network. Older drafts may say “w-dit”; it refers to the same concept.

This work explores w-bit as a specialized substrate for domains requiring fluid connectivity rather than precise arithmetic. Simulations indicate that w-bit fabrics can implement compositional non-linear logic (e.g., XOR) and effective gradient-based routing.

The architecture is best positioned as a **Neuromorphic Interconnect** or **Approximate Control Plane**—a programmable fabric where data packets navigate to destinations based on learned potential fields, rather than explicit address decoding. It is not intended to replace general-purpose CPU logic for serial, exact computation.

## 2. Formal Core

A w-dit network is defined by a graph of $N$ cells, where each cell $i$ can exist in one of $R$ states.

### 2.1 State Definition
In the physical (analog) realization, the state of cell $i$ is a probability distribution vector $\mathbf{s}_i \in \mathbb{R}^R$:
$$ \sum_{r=0}^{R-1} s_{i,r} = 1, \quad s_{i,r} \ge 0 $$

### 2.2 Parameters
The logic is encoded not in gates, but in weights:
1.  **Local Preference Vector** $\theta_i \in \mathbb{R}^R$: The intrinsic bias of cell $i$ toward specific states (e.g., input clamping).
2.  **Compatibility Matrix** $\Theta_{ij} \in \mathbb{R}^{R \times R}$: The pairwise influence, where $\Theta_{ij}(r, k)$ is the compatibility score added to state $r$ of cell $i$ if neighbor $j$ is in state $k$.

### 2.3 Energy / Compatibility
The system maximizes a global Compatibility function $H$ (equivalent to minimizing Energy $E = -H$):
$$ H(\mathbf{S}) = \sum_{i} \theta_i \cdot \mathbf{s}_i + \sum_{i,j} \mathbf{s}_i^\top \Theta_{ij} \mathbf{s}_j $$

### 2.4 Update Dynamics
The system evolves via a "Mean Field" relaxation. At each time step $t$, the input drive $\mathbf{u}_i$ for cell $i$ is:
$$ \mathbf{u}_i^{(t)} = \theta_i + \sum_{j \in \mathcal{N}(i)} \Theta_{ij} \mathbf{s}_j^{(t)} $$

**Stochastic Update (Temperature $T$):**
$$ \mathbf{s}_i^{(t+1)} = (1 - \alpha)\mathbf{s}_i^{(t)} + \alpha \cdot \text{Softmax}\left(\frac{\mathbf{u}_i^{(t)} + \mathbf{\epsilon}}{T}\right) $$
Where $\alpha$ is the integration rate (simulating capacitance) and $\mathbf{\epsilon} \sim \mathcal{N}(0, \sigma^2)$ represents analog noise.

## 3. Scope and Applicability

**Best Fit:**
*   **Routing Fabrics:** Massive parallel interconnects where packets self-route via potential fields.
*   **Sensor-Near Logic:** Low-precision classification or filtering directly on analog sensor outputs.
*   **Approximate Control:** Robust, decentralized decision-making in noisy environments.

**Not First Target:**
*   **Exact Arithmetic:** High-precision adders/multipliers (relaxation is too slow and imprecise).
*   **Serial Algorithmic Processing:** Traditional code execution (OS kernels, compilers).
*   **Storage-Intensive Tasks:** Standard DRAM/SRAM is far denser for pure storage.

## 4. Simulation Findings

### 4.1 Routing Efficiency (Experiment A)
A 10x10 w-dit grid successfully implemented a "Liquid Router."
*   **Mechanism:** Weights encoded a gradient potential field toward a target $T$.
*   **Result:** Packets navigated obstacles without explicit path planning algorithms.
*   **Latency:** Field stabilization required ~20 relaxation steps.
*   **Implication:** Offers a decentralized alternative for routing in mesh networks.

### 4.2 Noise Robustness (Experiment B)
An analog gating circuit (Transistor equivalent) was tested under high Gaussian noise ($\sigma=0.5$).
*   **Result:** System converged to correct logic states (>99% confidence) despite noise.
*   **Mechanism:** The relaxation dynamics act as a low-pass filter, suppressing transient noise.

### 4.3 Learning Capacity (Experiment C)
The architecture solved the XOR problem (non-linearly separable).
*   **Constraint:** Required $\ge 2$ hidden cells to form intermediate energy barriers.
*   **Performance:** Genetic optimization converged to $MSE < 0.02$.
*   **Conclusion:** Simulations demonstrate compositional non-linear decision capability in small w-dit networks, indicating the substrate can implement complex R-ary logic when sufficient intermediate cells are available.

## 5. Metric: Relaxation Cost Proxy (RCP)

To evaluate efficiency, we define **Relaxation Cost Proxy (RCP)**. This is an internal metric to track algorithmic cost, distinct from physical energy.

$$ RCP = \frac{N_{cells} \cdot N_{steps} \cdot C_{int}}{I_{out}} $$

Where:
*   $N_{cells}$: Active cells in the path.
*   $N_{steps}$: Cycles to reach stable state (Convergence Latency).
*   $C_{int}$: Cost of interaction (proportional to $R^2$ for dense connectivity).
*   $I_{out}$: Shannon information resolved at output (bits).

This metric penalizes slow convergence ($N_{steps}$) and excessive network size ($N_{cells}$), encouraging sparse, fast-settling designs.

## 6. Hardware Feasibility

The mathematical model ($\Theta_{ij} \mathbf{s}_j$) maps naturally to **Memristive Crossbar Arrays**, where multiplication occurs via Ohm's Law ($V=IR$). While this suggests a plausible substrate for implementation, any specific energy advantage over CMOS remains a hypothesis pending calibrated device parameters and physical prototyping.
