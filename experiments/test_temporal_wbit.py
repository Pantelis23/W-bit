import sys
import os
import random
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wbit.level3_temporal_wbit_network import Level3TemporalWbitNetwork

def rigorous_proof_temporal():
    print("=== PROOF OF ADAPTIVE LOGIC: Temporal Integration (Tau-bit) ===")
    
    noise_level = 5.0 
    trials = 2000
    R_max = 9
    
    # --- WZMA Structured Weight Patch ---
    centers = [1, 4, 7] 
    
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    unscaled_88 = sum(A[8][k] * B[8][k] for k in range(3))
    wzma_scale_factor = 3.0 / unscaled_88
    for i in range(R_max):
        for j in range(R_max):
            wzma_matrix[i][j] = sum(A[i][k] * B[j][k] for k in range(3)) * wzma_scale_factor
            
    def run_trials(description, use_beta, use_h_neuron, use_margin):
        success_count = 0
        for _ in range(trials):
            fabric = Level3TemporalWbitNetwork(num_cells=2, R_max=R_max, mode="adaptive")
            fabric.set_interaction_weights(1, 0, wzma_matrix)
            fabric.R_eff = 3
            
            # Force Cell 0 into State 8
            fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
            
            # Run for more steps because it takes time to integrate voltage
            for _ in range(15):
                fabric.step_temporal(temperature=0.5, noise_level=noise_level, dt=1.0,
                                     use_beta=use_beta, use_h_neuron=use_h_neuron, use_margin=use_margin)
                fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
                
            final_state = max(range(R_max), key=lambda idx: fabric.state[1][idx])
            
            if final_state == 7: # Target for R_eff=3
                success_count += 1
                
        rate = (success_count/trials)*100
        print(f"{description:45s} | Success: {success_count:4d}/{trials} ({rate:5.1f}%)")
        return rate

    print(f"\n[Test parameters: Noise={noise_level}, Signal Peak={wzma_matrix[8][8]:.2f}, Rank=3, Trials={trials}]")

    print("\n--- ABLATION WITH TRUE TEMPORAL PHYSICS ---")
    run_trials("1. Temporal Baseline (No ML Mechanisms)", False, False, False)
    run_trials("2. Active Leak Only (H-Neurons)", False, True, False)
    run_trials("3. Threshold Sensitization Only (KV Beta)", True, False, False)
    run_trials("4. Voltage Margin Only (Calibration)", False, False, True)
    print("-" * 65)
    run_trials("5. ALL TEMPORAL MECHANISMS COMBINED", True, True, True)

if __name__ == "__main__":
    rigorous_proof_temporal()
