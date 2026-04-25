import sys
import os
import random
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wbit.level3_wbit_network import Level3WbitNetwork

def rigorous_proof():
    print("=== PROOF OF ADAPTIVE LOGIC: Tuned Synthesis v4 ===")
    
    noise_level = 5.0 
    trials = 1000
    R_max = 9
    
    # --- Tuned Hyperparameters from Optuna ---
    beta_multiplier = 1.1270
    h_neuron_penalty = 9.1883
    calibration_margin = 1.8786
    temp_spike = 8.1613
    wzma_scale = 13.6420
    
    centers = [1, 4, 7] 
    
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    for i in range(R_max):
        for j in range(R_max):
            dot = sum(A[i][k] * B[j][k] for k in range(3))
            wzma_matrix[i][j] = dot * wzma_scale
            
    def run_trials(R_eff_target):
        success_count = 0
        for _ in range(trials):
            fabric = Level3WbitNetwork(num_cells=2, R_max=R_max, mode="adaptive")
            fabric.set_interaction_weights(1, 0, wzma_matrix)
            fabric.R_eff = R_eff_target
            
            # Inject hyperparams
            original_step = fabric.step
            def tuned_step(temperature=0.5, noise_level=noise_level, dt=1.0):
                new_states = []
                for i in range(fabric.num_cells):
                    input_drive = list(fabric.theta[i])
                    for (target, source), matrix in fabric.Theta.items():
                        if target == i:
                            source_state = fabric.state[source]
                            for r_i in range(fabric.R_max):
                                dot_prod = sum(matrix[r_i][r_j] * source_state[r_j] for r_j in range(fabric.R_max))
                                input_drive[r_i] += dot_prod
                    if noise_level > 0:
                        for r in range(fabric.R_max):
                            input_drive[r] += random.gauss(0, noise_level)
                    
                    if fabric.R_eff < fabric.R_max:
                        bin_size = fabric.R_max / fabric.R_eff
                        allowed_centers = [min(int((b + 0.5) * bin_size), fabric.R_max - 1) for b in range(fabric.R_eff)]
                        for r in range(fabric.R_max):
                            if r in allowed_centers:
                                input_drive[r] += math.log(bin_size) * beta_multiplier
                            else:
                                input_drive[r] -= h_neuron_penalty
                    
                    sorted_drives = sorted(input_drive, reverse=True)
                    if (sorted_drives[0] - sorted_drives[1]) < calibration_margin:
                        temperature *= temp_spike
                    
                    target_dist = fabric.softmax(input_drive, temperature)
                    current_dist = fabric.state[i]
                    updated_dist = [current_dist[r] + dt * (target_dist[r] - current_dist[r]) for r in range(fabric.R_max)]
                    new_states.append(fabric._project_state(updated_dist))
                fabric.state = new_states
            fabric.step = tuned_step
            
            fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
            for _ in range(5):
                fabric.step()
                fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
                
            final_state = max(range(R_max), key=lambda idx: fabric.state[1][idx])
            target_state = 8 if R_eff_target == 9 else 7
            if final_state == target_state:
                success_count += 1
                
        return success_count

    print(f"\n[Test parameters: Noise={noise_level}, Signal Peak={wzma_matrix[8][8]:.2f}, Rank=3, Trials={trials}]")

    print("\n[Test 1: Fixed High-Precision (R=9) + WZMA Weights]")
    success_R9 = run_trials(9)
    print(f"Success Rate: {success_R9}/{trials} ({(success_R9/trials)*100:.1f}%)")
    
    print("\n[Test 2: Adaptive Phi-Mode (R=3) + Tuned Physics]")
    success_R3 = run_trials(3)
    print(f"Success Rate: {success_R3}/{trials} ({(success_R3/trials)*100:.1f}%)")

if __name__ == "__main__":
    rigorous_proof()
