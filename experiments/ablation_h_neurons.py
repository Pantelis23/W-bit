import sys
import os
import random
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wbit.level3_wbit_network import Level3WbitNetwork

def ablation_test():
    print("=== ABLATION STUDY: Removing H-Neuron Suppression ===")
    
    # Tuned Hyperparameters (Keep everything else exactly the same)
    beta_multiplier = 1.1270
    calibration_margin = 1.8786
    temp_spike = 8.1613
    wzma_scale = 13.6420
    
    trials = 2000
    R_max = 9
    centers = [1, 4, 7] 
    
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    for i in range(R_max):
        for j in range(R_max):
            dot = sum(A[i][k] * B[j][k] for k in range(3))
            wzma_matrix[i][j] = dot * wzma_scale
            
    def run_scenario(noise_level, description, use_h_penalty):
        success_count = 0
        h_neuron_penalty = 9.1883 if use_h_penalty else 0.0 # <--- ABLATION SWITCH
        
        for _ in range(trials):
            fabric = Level3WbitNetwork(num_cells=2, R_max=R_max, mode="adaptive")
            fabric.set_interaction_weights(1, 0, wzma_matrix)
            fabric.R_eff = 3 
            
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
                                input_drive[r] -= h_neuron_penalty # <--- IF 0.0, THIS DOES NOTHING
                    
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
            if final_state == 7:
                success_count += 1
                
        rate = (success_count/trials)*100
        print(f"  {description}: {success_count}/{trials} ({rate:.1f}%)")
        return rate

    print("\n[Noise Level: 5.0]")
    run_scenario(5.0, "WITH H-Neuron Suppression (Baseline)   ", True)
    run_scenario(5.0, "WITHOUT H-Neuron Suppression (Ablated)", False)

if __name__ == "__main__":
    ablation_test()
