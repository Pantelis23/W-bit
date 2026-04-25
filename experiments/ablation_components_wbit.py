import sys
import os
import random
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wbit.level3_wbit_network import Level3WbitNetwork

def run_ablation_study():
    print("=== ABLATION STUDY: Isolating Level 3 Mechanisms ===")
    
    noise_level = 5.0
    trials = 2000
    R_max = 9
    
    # --- Matrix Definitions ---
    # 1. Base Diagonal Matrix (Standard "Copy" logic, highly brittle)
    base_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    for i in range(R_max):
        base_matrix[i][i] = 3.0 # WEAK SIGNAL
        
    # 2. WZMA Structured Matrix (Low-rank, smoothed landscape)
    centers = [1, 4, 7] 
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    
    # Calculate the unscaled value at [8][8] to find the normalization factor
    unscaled_88 = sum(A[8][k] * B[8][k] for k in range(3))
    wzma_scale_factor = 3.0 / unscaled_88 # Force [8][8] to be exactly 3.0
    
    for i in range(R_max):
        for j in range(R_max):
            wzma_matrix[i][j] = sum(A[i][k] * B[j][k] for k in range(3)) * wzma_scale_factor
            
    def run_scenario(description, use_wzma, use_beta, use_margin):
        success_count = 0
        
        # Hyperparameters
        beta_multiplier = 1.1270 if use_beta else 0.0
        calibration_margin = 1.8786 if use_margin else 0.0
        temp_spike = 8.1613 if use_margin else 1.0
        
        interaction_matrix = wzma_matrix if use_wzma else base_matrix

        for _ in range(trials):
            fabric = Level3WbitNetwork(num_cells=2, R_max=R_max, mode="adaptive")
            fabric.set_interaction_weights(1, 0, interaction_matrix)
            fabric.R_eff = 3 
            
            # Custom step function to inject specific mechanisms
            def tuned_step(temperature=0.5, noise_level=noise_level, dt=1.0):
                new_states = []
                for i in range(fabric.num_cells):
                    input_drive = list(fabric.theta[i])
                    
                    # Apply Matrix
                    for (target, source), matrix in fabric.Theta.items():
                        if target == i:
                            source_state = fabric.state[source]
                            for r_i in range(fabric.R_max):
                                input_drive[r_i] += sum(matrix[r_i][r_j] * source_state[r_j] for r_j in range(fabric.R_max))
                    
                    # Inject Noise
                    if noise_level > 0:
                        for r in range(fabric.R_max):
                            input_drive[r] += random.gauss(0, noise_level)
                    
                    # KV Beta Bias Mechanism
                    if fabric.R_eff < fabric.R_max and use_beta:
                        bin_size = fabric.R_max / fabric.R_eff
                        allowed_centers = [min(int((b + 0.5) * bin_size), fabric.R_max - 1) for b in range(fabric.R_eff)]
                        for r in allowed_centers:
                            input_drive[r] += math.log(bin_size) * beta_multiplier
                    
                    # Calibration Margin Mechanism
                    if use_margin:
                        sorted_drives = sorted(input_drive, reverse=True)
                        if (sorted_drives[0] - sorted_drives[1]) < calibration_margin:
                            temperature *= temp_spike
                    
                    target_dist = fabric.softmax(input_drive, temperature)
                    current_dist = fabric.state[i]
                    updated_dist = [current_dist[r] + dt * (target_dist[r] - current_dist[r]) for r in range(fabric.R_max)]
                    new_states.append(fabric._project_state(updated_dist))
                fabric.state = new_states
                
            fabric.step = tuned_step
            
            # Transmit State 8
            fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
            for _ in range(5):
                fabric.step()
                fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
                
            final_state = max(range(R_max), key=lambda idx: fabric.state[1][idx])
            if final_state == 7: # Target for R_eff=3
                success_count += 1
                
        rate = (success_count/trials)*100
        print(f"{description:40s} | Success: {success_count:4d}/{trials} ({rate:5.1f}%)")
        return rate

    print(f"\n[Parameters: Noise={noise_level}, Signal={base_matrix[0][0]:.2f}, R_eff=3, Trials={trials}]")
    print("-" * 65)
    
    base = run_scenario("1. Baseline (No Mechanisms)", False, False, False)
    
    print("-" * 65)
    wzma = run_scenario("2. WZMA Structured Matrix Only", True, False, False)
    beta = run_scenario("3. KV Beta-Bias Only", False, True, False)
    marg = run_scenario("4. Calibration Margin Only", False, False, True)
    
    print("-" * 65)
    wzma_beta = run_scenario("5. WZMA + KV Beta-Bias", True, True, False)
    wzma_marg = run_scenario("6. WZMA + Calibration Margin", True, False, True)
    beta_marg = run_scenario("7. KV Beta-Bias + Calibration Margin", False, True, True)
    
    print("-" * 65)
    full = run_scenario("8. ALL MECHANISMS COMBINED", True, True, True)

if __name__ == "__main__":
    run_ablation_study()