# W-bit Level 3: The Base Primitive

## Core Foundation (The Immutable Base)
The W-bit is a radix-$R$ logic primitive. 
*   **State:** $s \in \{0,1,\dots,R-1\}$
*   **Energy Model:** $E(\mathbf{s}) = \sum_i \theta_i(s_i) + \sum_{i<j} \Theta_{ij}(s_i, s_j)$
*   **Computation:** The system settles into low-energy / high-compatibility states via $\arg\max$ or softmax.
*   **Level 2:** The logic is an energy/compatibility model over $R$-ary states.

## Defining Level 3 (Self-Modification of the Base)
Level 3 means the fabric edits its own rules. 
If Level 2 is "settling into the energy landscape," Level 3 is **the system actively reprogramming its own $\theta$ (local preference) and $\Theta$ (pairwise compatibility) during runtime to restructure the energy landscape itself.**

The $\phi$-bit, $\tau$-bit, and Adaptive $R_{effective}$ concepts are *secondary tools* to be deployed only if the base Level 3 self-modification encounters specific physical or noise-related roadblocks.