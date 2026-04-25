import sys
import os
import random
import math
import itertools

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

    def step_spatial(self, noise_level, dt, use_beta, use_h_neuron, use_margin, 
                     beta_multiplier=1.1270, h_neuron_penalty=9.1883, 
                     calibration_margin=1.8786, temp_spike=8.1613):
        temperature = 0.5
        new_states = []
        for i in range(self.num_cells):
            input_drive = list(self.theta[i])
            for (target, source), matrix in self.Theta.items():
                if target == i:
                    source_state = self.state[source]
                    for r_i in range(self.R_max):
                        input_drive[r_i] += sum(matrix[r_i][r_j] * source_state[r_j] for r_j in range(self.R_max))
            
            if noise_level > 0:
                for r in range(self.R_max):
                    input_drive[r] += random.gauss(0, noise_level)
            
            bin_size = self.R_max / self.R_eff
            allowed_centers = [min(int((b + 0.5) * bin_size), self.R_max - 1) for b in range(self.R_eff)]
            
            for r in range(self.R_max):
                if r in allowed_centers:
                    if use_beta:
                        input_drive[r] += math.log(bin_size) * beta_multiplier
                else:
                    if use_h_neuron:
                        input_drive[r] -= h_neuron_penalty
                        
            if use_margin:
                sorted_drives = sorted(input_drive, reverse=True)
                if (sorted_drives[0] - sorted_drives[1]) < calibration_margin:
                    temperature *= temp_spike
                    
            target_dist = self.softmax(input_drive, temperature)
            current_dist = self.state[i]
            updated_dist = [current_dist[r] + dt * (target_dist[r] - current_dist[r]) for r in range(self.R_max)]
            new_states.append(self._project_state(updated_dist))
        self.state = new_states

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
                        input_current[r_i] += sum(matrix[r_i][r_j] * source_state[r_j] for r_j in range(self.R_max))
                        
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
                        target_dist = [1.0 if x == r else 0.0 for x in range(self.R_max)]
                        self.V_m[i] = [0.0 for _ in range(self.R_max)]
                        break
            else:
                target_dist = self.softmax(self.V_m[i], temperature)
                
            new_states.append(self._project_state(target_dist))
        self.state = new_states

def run_exhaustive_ablation():
    print("=== EXHAUSTIVE ABLATION: All Combinations Across All Modes ===")
    noise_level = 5.0
    trials = 1000
    R_max = 9
    
    # Matrices
    base_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    for i in range(R_max): base_matrix[i][i] = 3.0 
        
    centers = [1, 4, 7] 
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    unscaled_88 = sum(A[8][k] * B[8][k] for k in range(3))
    for i in range(R_max):
        for j in range(R_max):
            wzma_matrix[i][j] = sum(A[i][k] * B[j][k] for k in range(3)) * (3.0 / unscaled_88)

    matrices = {"Base": base_matrix, "WZMA": wzma_matrix}
    modes = ["Spatial", "Temporal"]
    
    # combinations: (use_beta, use_h_neuron, use_margin)
    combinations = list(itertools.product([False, True], repeat=3)) 
    
    results = []
    
    for mode in modes:
        for mat_name, mat in matrices.items():
            for use_beta, use_h, use_margin in combinations:
                success_count = 0
                for _ in range(trials):
                    fabric = UnifiedTestFabric(num_cells=2, R_max=R_max)
                    fabric.set_interaction_weights(1, 0, mat)
                    
                    # Force Cell 0 into State 8
                    fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
                    
                    steps = 5 if mode == "Spatial" else 15
                    for _ in range(steps):
                        if mode == "Spatial":
                            fabric.step_spatial(noise_level, 1.0, use_beta, use_h, use_margin)
                        else:
                            fabric.step_temporal(noise_level, 1.0, use_beta, use_h, use_margin)
                        fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
                        
                    final_state = max(range(R_max), key=lambda idx: fabric.state[1][idx])
                    if final_state == 7: # 7 is the target center for 8 in phi-mode
                        success_count += 1
                        
                rate = (success_count/trials)*100
                name = f"{mode} | {mat_name} | Beta:{int(use_beta)} H:{int(use_h)} Marg:{int(use_margin)}"
                results.append((rate, name))
                print(f"{name:50s} : {rate:5.1f}%")

    print("\n=== TOP 10 COMBINATIONS ===")
    results.sort(reverse=True, key=lambda x: x[0])
    for rate, name in results[:10]:
        print(f"{name:50s} : {rate:5.1f}%")

if __name__ == "__main__":
    run_exhaustive_ablation()