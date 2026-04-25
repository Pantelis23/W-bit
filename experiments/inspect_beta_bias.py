import sys
import os
import random
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wbit.level3_wbit_network import Level3WbitNetwork

def inspect_beta():
    print("=== KV BETA-BIAS: Deep Inspection ===\n")
    
    noise_level = 5.0
    trials = 5000 # High sample size to detect edge cases
    R_max = 9
    
    # Tuned hyperparams
    beta_multiplier = 1.1270
    calibration_margin = 1.8786
    temp_spike = 8.1613
    
    # WZMA Matrix (Peak 3.0)
    centers = [1, 4, 7] 
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    unscaled_88 = sum(A[8][k] * B[8][k] for k in range(3))
    wzma_scale = 3.0 / unscaled_88
    for i in range(R_max):
        for j in range(R_max):
            wzma_matrix[i][j] = sum(A[i][k] * B[j][k] for k in range(3)) * wzma_scale
            
    def run_inspector(use_beta):
        success = 0
        failure_states = {r: 0 for r in range(R_max)}
        margin_triggers = 0
        
        for _ in range(trials):
            fabric = Level3WbitNetwork(num_cells=2, R_max=R_max, mode="adaptive")
            fabric.set_interaction_weights(1, 0, wzma_matrix)
            fabric.R_eff = 3 
            
            def tuned_step(temperature=0.5, noise_level=noise_level, dt=1.0):
                nonlocal margin_triggers
                new_states = []
                for i in range(fabric.num_cells):
                    input_drive = list(fabric.theta[i])
                    for (target, source), matrix in fabric.Theta.items():
                        if target == i:
                            source_state = fabric.state[source]
                            for r_i in range(fabric.R_max):
                                input_drive[r_i] += sum(matrix[r_i][r_j] * source_state[r_j] for r_j in range(fabric.R_max))
                    if noise_level > 0:
                        for r in range(fabric.R_max):
                            input_drive[r] += random.gauss(0, noise_level)
                    
                    if fabric.R_eff < fabric.R_max and use_beta:
                        bin_size = fabric.R_max / fabric.R_eff
                        allowed_centers = [min(int((b + 0.5) * bin_size), fabric.R_max - 1) for b in range(fabric.R_eff)]
                        for r in allowed_centers:
                            input_drive[r] += math.log(bin_size) * beta_multiplier
                    
                    sorted_drives = sorted(input_drive, reverse=True)
                    if (sorted_drives[0] - sorted_drives[1]) < calibration_margin:
                        temperature *= temp_spike
                        if i == 1: # Only track cell 1's triggers
                            margin_triggers += 1
                    
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
                success += 1
            else:
                failure_states[final_state] += 1
                
        return success, failure_states, margin_triggers / (trials * 5) # avg triggers per step

    print("Running WZMA + Margin (No Beta)...")
    s_no_beta, f_no_beta, mt_no_beta = run_inspector(False)
    
    print("Running WZMA + Margin + Beta...")
    s_beta, f_beta, mt_beta = run_inspector(True)
    
    print("\n--- RESULTS ---")
    print(f"NO BETA:   Success: {s_no_beta}/{trials} ({s_no_beta/trials*100:.1f}%) | Avg Margin Triggers: {mt_no_beta:.2f}/step")
    print(f"WITH BETA: Success: {s_beta}/{trials} ({s_beta/trials*100:.1f}%) | Avg Margin Triggers: {mt_beta:.2f}/step")
    
    print("\n--- FAILURE DISTRIBUTION (Where does the signal go?) ---")
    print("State | No Beta | With Beta")
    for r in [1, 4]: # The other valid phi-mode centers
        print(f"  {r}   | {f_no_beta[r]:7d} | {f_beta[r]:9d}")

if __name__ == "__main__":
    inspect_beta()
