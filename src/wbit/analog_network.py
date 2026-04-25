import random
import math

class AnalogWbitNetwork:
    def __init__(self, num_cells, R, mode="wbit"):
        """
        Initialize the Analog W-bit Network.
        
        State is now a continuous vector of size (N, R).
        Each cell 'i' has a distribution over R states.
        
        Aeternum AET Mode Compliance:
        If mode="adaptive", R must correspond to a valid level n in {1, 2, 4, 8, 16}.
        R = 2n + 1.
        Valid R: {3, 5, 9, 17, 33}.
        """
        self.num_cells = num_cells
        self.mode = mode
        
        # Aeternum AET Parameters
        self.n = 0
        self.w = 0
        self.L = 0
        self.sat_count = 0
        
        if self.mode == "adaptive":
            # Reverse calculate n from R
            # R = 2n + 1 => n = (R - 1) / 2
            n_calc = (R - 1) / 2
            if n_calc not in [1, 2, 4, 8, 16]:
                # If R is not exact, find closest valid n
                valid_ns = [1, 2, 4, 8, 16]
                n_calc = min(valid_ns, key=lambda x: abs(x - n_calc))
                R = 2 * n_calc + 1
                print(f"[Adaptive] Adjusted R to {R} (n={n_calc}) for Aeternum compliance.")
            
            self.n = int(n_calc)
            self.R = R
            self.w = math.ceil(math.log2(self.R))
            self.L = 32 // self.w
        else:
            self.R = R
        
        # State: list of lists (N, R), initialized to random small noise (neutral)
        # Represents voltage/activation for each candidate state.
        if self.mode == "binary":
            self.state = [[random.uniform(0.4, 0.6) for _ in range(self.R)] for _ in range(num_cells)]
        else:
            self.state = [[random.uniform(0.4, 0.6) for _ in range(self.R)] for _ in range(num_cells)]
        # Normalize initial state to look like probabilities
        for i in range(num_cells):
            s_sum = sum(self.state[i])
            self.state[i] = [x / s_sum for x in self.state[i]]
        
        # Local preference (bias) theta: list of lists (N, R)
        self.theta = [[0.0] * self.R for _ in range(num_cells)]
        
        # Pairwise compatibility Theta: Dictionary (i, j) -> [[R x R]]
        # Weights matrix connecting cell j (columns) to cell i (rows).
        self.Theta = {}

    def get_aet_stats(self):
        """Return Aeternum-specific stats if in adaptive mode."""
        if self.mode != "adaptive":
            return {}
        return {
            "n": self.n,
            "w": self.w,
            "L": self.L,
            "sat_count": self.sat_count,
            "value_range": f"[-{self.n}, +{self.n}]"
        }

    def set_local_weights(self, cell_idx, weights):
        if len(weights) != self.R:
            raise ValueError(f"Weights must be of length {self.R}")
        self.theta[cell_idx] = list(weights)

    def reset_local_weights(self):
        self.theta = [[0.0] * self.R for _ in range(self.num_cells)]

    def set_interaction_weights(self, i, j, weights_matrix):
        """
        Set weights from cell j to cell i.
        weights_matrix[r_i][r_j] is weight from state r_j of cell j to state r_i of cell i.
        """
        if len(weights_matrix) != self.R or any(len(row) != self.R for row in weights_matrix):
            raise ValueError(f"Interaction matrix must be {self.R}x{self.R}")
        self.Theta[(i, j)] = [list(row) for row in weights_matrix]

    def softmax(self, x, temperature):
        # Shift for stability
        m = max(x)
        shifted = [val - m for val in x]
        exps = [math.exp(val / temperature) for val in shifted]
        s = sum(exps)
        return [e / s for e in exps]

    def step(self, temperature=1.0, noise_level=0.0, dt=0.5):
        """
        Perform a soft update (Euler integration-like).
        
        Args:
            temperature: Controls sharpness of decision (lower = sharper).
            noise_level: Magnitude of random noise added to inputs.
            dt: 'Time step' or learning rate (0.0 to 1.0). 
                1.0 = instant jump to new value.
                0.1 = slow smoothing.
        """
        new_states = []
        
        for i in range(self.num_cells):
            # 1. Calculate Input Drive
            # Start with local bias
            input_drive = list(self.theta[i])
            
            # Add inputs from neighbors
            for (target, source), matrix in self.Theta.items():
                if target == i:
                    source_state = self.state[source]
                    # Matrix-Vector multiplication: matrix * source_state
                    for r_i in range(self.R):
                        dot_prod = 0.0
                        for r_j in range(self.R):
                            dot_prod += matrix[r_i][r_j] * source_state[r_j]
                        input_drive[r_i] += dot_prod
            
            # 2. Add Noise (Simulate analog imperfections)
            if noise_level > 0:
                for r in range(self.R):
                    input_drive[r] += random.gauss(0, noise_level)
            
            # 3. Activation (Softmax represents settling into a state)
            target_dist = self.softmax(input_drive, temperature)
            
            # Aeternum AET Saturation Tracking (Adaptive Mode)
            if self.mode == "adaptive":
                self._check_aet_saturation(target_dist)

            # 4. Update state with inertia (Low Pass Filter behavior)
            # current + dt * (target - current)
            current_dist = self.state[i]
            updated_dist = []
            for r in range(self.R):
                val = current_dist[r] + dt * (target_dist[r] - current_dist[r])
                updated_dist.append(val)
            if self.mode == "binary":
                best = max(range(self.R), key=lambda idx: updated_dist[idx])
                bin_state = [0.0] * self.R
                bin_state[best] = 1.0
                new_states.append(bin_state)
            else:
                new_states.append(updated_dist)
            
        self.state = new_states

    def _check_aet_saturation(self, dist):
        """
        Check for saturation in AET mode.
        Aeternum Saturation: Value hits -n (index 0) or +n (index R-1).
        In analog probabilistic terms, we count it if the probability mass 
        is heavily concentrated (>0.9) at the boundaries.
        """
        p_min = dist[0]
        p_max = dist[-1]
        threshold = 0.9
        
        if p_min > threshold or p_max > threshold:
            self.sat_count += 1

    def run_until_stable(self, max_steps=100, tolerance=1e-3, temperature=0.5, noise=0.0):
        for step in range(max_steps):
            old_state = [list(s) for s in self.state]
            self.step(temperature=temperature, noise_level=noise)
            
            # Check delta
            max_delta = 0.0
            for i in range(self.num_cells):
                for r in range(self.R):
                    max_delta = max(max_delta, abs(self.state[i][r] - old_state[i][r]))
            
            if max_delta < tolerance:
                return step # Converged
        return max_steps

    def get_hard_state(self):
        """Return the discrete state (argmax) for visualization."""
        hard_states = []
        for i in range(self.num_cells):
            # argmax
            best_r = 0
            best_val = self.state[i][0]
            for r in range(1, self.R):
                if self.state[i][r] > best_val:
                    best_val = self.state[i][r]
                    best_r = r
            hard_states.append(best_r)
        return hard_states

    def calculate_RCP(self, n_steps, i_out=1.0, c_int=None, n_active_cells=None):
        """
        Calculate Relaxation Cost Proxy (RCP).
        RCP = (n_active_cells * n_steps * C_int) / I_out
        
        Args:
            n_steps (int): Number of steps taken to converge.
            i_out (float): Information bits at output (default 1.0).
            c_int (float): Cost of interaction per cell. Defaults to R^2.
            n_active_cells (int): Number of active cells. Defaults to self.num_cells.
        """
        if c_int is None:
            c_int = self.R * self.R
            
        if n_active_cells is None:
            n_active_cells = self.num_cells
            
        return (n_active_cells * n_steps * c_int) / i_out

    def __repr__(self):
        # Print readable state vectors
        s = "AnalogWbitNetwork State:\n"
        for i, st in enumerate(self.state):
            formatted = [f"{x:.2f}" for x in st]
            s += f"  Cell {i}: {formatted}\n"
        return s
