import sys
import os
import random
import math
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wbit.analog_network import AnalogWbitNetwork

class UnifiedTestFabric(AnalogWbitNetwork):
    def __init__(self, num_cells, R_max=9):
        super().__init__(num_cells, R=R_max, mode="adaptive")
        self.R_max = self.R
        self.R_eff = 3 # Phi-mode
        self.V_m = [[0.0 for _ in range(self.R_max)] for _ in range(num_cells)]
        
    def _project_state(self, state_dist):
        projected = [0.0] * self.R_max
        bin_size = self.R_max / self.R_eff
        for r in range(self.R_max):
            bin_idx = min(int(r / bin_size), self.R_eff - 1)
            center_r = int((bin_idx + 0.5) * bin_size)
            center_r = min(center_r, self.R_max - 1)
            projected[center_r] += state_dist[r]
        return projected

    def step_temporal(self, noise_level, dt, use_beta, use_h_neuron, use_margin,
                      tau_leak=5.0, tau_h_leak=1.0, V_threshold=10.0, beta_sensitization=0.8):
        temperature = 0.5
        new_states = []
        for i in range(self.num_cells):
            input_current = list(self.theta[i])
            for (target, source), matrix in self.Theta.items():
                if target == i:
                    source_state = self.state[source]
                    for r_i in range(self.R_max):
                        input_current[r_i] += sum(matrix[r_i][r_j] * source_state[r_j] for r_j in self.R_max_range)
                        
            if noise_level > 0:
                for r in range(self.R_max):
                    input_current[r] += random.gauss(0, noise_level)
                    
            bin_size = self.R_max / self.R_eff
            allowed_centers = [min(int((b + 0.5) * bin_size), self.R_max - 1) for b in range(self.R_eff)]
            
            for r in range(self.R_max):
                current_tau = tau_h_leak if (use_h_neuron and r not in allowed_centers) else tau_leak
                dv = -(self.V_m[i][r] / current_tau) * dt
                self.V_m[i][r] = max(0.0, self.V_m[i][r] + dv + input_current[r] * dt)
                
            target_dist = list(self.state[i])
            
            if use_margin:
                for r in range(self.R_max):
                    local_threshold = V_threshold * beta_sensitization if (use_beta and r in allowed_centers) else V_threshold
                    if self.V_m[i][r] >= local_threshold:
                        target_dist = [1.0 if x == r else 0.0 for x in self.R_max_range]
                        self.V_m[i] = [0.0 for _ in self.R_max_range]
                        break
            else:
                target_dist = self.softmax(self.V_m[i], temperature)
                
            new_states.append(self._project_state(target_dist))
        self.state = new_states

def million_cycle_test():
    print("=== MILLION CYCLE STRESS TEST: The Top 4 Temporal Configurations ===")
    noise_level = 5.0
    trials = 1_000_000 # The requested million cycles
    R_max = 9
    
    centers = [1, 4, 7] 
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    unscaled_88 = sum(A[8][k] * B[8][k] for k in range(3))
    for i in range(R_max):
        for j in range(R_max):
            wzma_matrix[i][j] = sum(A[i][k] * B[j][k] for k in range(3)) * (3.0 / unscaled_88)

    configs = [
        ("Beta:0 H:0", False, False),
        ("Beta:1 H:1", True, True),
        ("Beta:0 H:1", False, True),
        ("Beta:1 H:0", True, False)
    ]
    
    for name, use_beta, use_h in configs:
        success_count = 0
        start_time = time.time()
        
        # To optimize for a million cycles in Python, we lift some variables out of the loop
        # and pre-allocate the fabric
        fabric = UnifiedTestFabric(num_cells=2, R_max=R_max)
        fabric.set_interaction_weights(1, 0, wzma_matrix)
        fabric.R_max_range = range(R_max)
        
        # We will run the simulation in batches to avoid restarting the object 1M times.
        # We just re-inject the state 8 signal continuously and track how often cell 1 holds state 7.
        # This simulates a continuous, million-step physical operation.
        
        fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
        
        # Let it settle for 10 steps before we start counting
        for _ in range(10):
            fabric.step_temporal(noise_level, 1.0, use_beta, use_h, False)
            fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
            
        print(f"\nStarting {name} (1,000,000 continuous integration steps)...")
        
        for _ in range(trials):
            fabric.step_temporal(noise_level, 1.0, use_beta, use_h, False)
            fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
            
            # Since we are using softmax (Marg:0), we check the argmax
            final_state = max(range(R_max), key=lambda idx: fabric.state[1][idx])
            if final_state == 7:
                success_count += 1
                
        elapsed = time.time() - start_time
        rate = (success_count/trials)*100
        print(f"  Result: {success_count}/{trials} ({rate:5.2f}%) | Time: {elapsed:.1f}s")

if __name__ == "__main__":
    million_cycle_test()
