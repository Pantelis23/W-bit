import sys
import os
import math
import random
import optuna

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wbit.level3_wbit_network import Level3WbitNetwork

def objective(trial):
    # Hyperparameters to tune
    beta_multiplier = trial.suggest_float("beta_multiplier", 0.1, 5.0)
    h_neuron_penalty = trial.suggest_float("h_neuron_penalty", 1.0, 20.0)
    calibration_margin = trial.suggest_float("calibration_margin", 0.1, 3.0)
    temp_spike = trial.suggest_float("temp_spike", 1.1, 10.0)
    wzma_scale = trial.suggest_float("wzma_scale", 1.0, 15.0)
    
    noise_level = 5.0 
    trials = 200
    R_max = 9
    
    # --- WZMA Structured Weight Patch ---
    centers = [1, 4, 7]
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    for i in range(R_max):
        for j in range(R_max):
            dot = sum(A[i][k] * B[j][k] for k in range(3))
            wzma_matrix[i][j] = dot * wzma_scale
            
    success_count = 0
    for _ in range(trials):
        fabric = Level3WbitNetwork(num_cells=2, R_max=R_max, mode="adaptive")
        fabric.set_interaction_weights(1, 0, wzma_matrix)
        fabric.R_eff = 3 # Force Phi-mode
        
        # Inject hyperparams into the step logic (Monkey-patching for the trial)
        original_step = fabric.step
        
        def tuned_step(temperature=0.5, noise_level=None, dt=1.0):
            if noise_level is None: noise_level = fabric.noise_estimate
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
                
                bin_size = fabric.R_max / fabric.R_eff
                allowed_centers = [min(int((b + 0.5) * bin_size), fabric.R_max - 1) for b in range(fabric.R_eff)]
                
                for r in range(fabric.R_max):
                    if r in allowed_centers:
                        beta_bias = math.log(bin_size) * beta_multiplier
                        input_drive[r] += beta_bias
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
        
        # Force Cell 0 into State 8
        fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
        
        for _ in range(5):
            fabric.step(temperature=0.5, noise_level=noise_level, dt=1.0)
            fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R_max)]
            
        final_state = max(range(R_max), key=lambda idx: fabric.state[1][idx])
        if final_state == 7: # Target for State 8 in R=3 mode
            success_count += 1
            
    return success_count / trials

if __name__ == "__main__":
    print("=== TUNING W-BIT LEVEL 3 PARAMETERS ===")
    study = optuna.create_study(direction="maximize")
    # Reduced trials for speed, but enough to find the optimal ratio
    study.optimize(objective, n_trials=30) 
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Success Rate: {trial.value * 100:.1f}%")
    print("  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value:.4f}")