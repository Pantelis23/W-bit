import sys
import os
import copy
import random
import math

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit.analog_network import AnalogWbitNetwork

def evaluate_analog(net, inputs, targets, input_indices, output_index, max_steps=30):
    """
    Evaluates the Analog Network on a batch of inputs/targets.
    Returns: Mean Squared Error (Lower is better) and Accuracy (Argmax).
    """
    R = net.R
    mse_total = 0.0
    correct_count = 0
    
    for input_vals, target_val in zip(inputs, targets):
        # 1. Reset Local Biases
        net.reset_local_weights()
        
        # 2. Clamp Inputs (Strong Voltage Bias)
        for idx, val in zip(input_indices, input_vals):
            bias = [-5.0] * R
            bias[val] = 10.0
            net.set_local_weights(idx, bias)
            
        # 3. Reset State (Random Neutral)
        net.state = [[random.uniform(0.4, 0.6) for _ in range(R)] for _ in range(net.num_cells)]
        # Normalize
        for i in range(net.num_cells):
            s = sum(net.state[i])
            net.state[i] = [x/s for x in net.state[i]]
            
        # 4. Run Physics Simulation
        # Using same params as the successful router demo
        net.run_until_stable(max_steps=max_steps, temperature=0.3, noise=0.1)
        
        # 5. Measure Error
        output_dist = net.state[output_index]
        
        # Target distribution (One-hot)
        target_dist = [0.0] * R
        target_dist[target_val] = 1.0
        
        # MSE
        err = sum((o - t)**2 for o, t in zip(output_dist, target_dist))
        mse_total += err
        
        # Accuracy (Argmax)
        predicted = output_dist.index(max(output_dist))
        if predicted == target_val:
            correct_count += 1
            
    return mse_total / len(inputs), correct_count / len(inputs)

def add_noise_to_matrix(matrix, scale):
    for r in range(len(matrix)):
        for c in range(len(matrix[r])):
            matrix[r][c] += random.gauss(0, scale)

def add_noise_to_vector(vector, scale):
    for i in range(len(vector)):
        vector[i] += random.gauss(0, scale)

def train_analog_xor():
    print("=== Analog W-bit Training: The XOR Problem ===")
    print("Objective: Train a continuous physical system to solve XOR.")
    print("Why XOR? It requires non-linear separability (hidden layers/states).")
    
    R = 2
    # Cells: Inputs A, B. Output Y. Hidden H.
    # We need a hidden cell because XOR is not linearly separable.
    # Network: A, B -> H1, H2 -> Y (and maybe direct A->Y, B->Y)
    A, B, H1, H2, Y = 0, 1, 2, 3, 4
    num_cells = 5
    
    # Truth Table for XOR
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    targets = [0, 1, 1, 0]
    
    # Initialize Network
    best_net = AnalogWbitNetwork(num_cells, R)
    
    # Initialize random small weights for all relevant connections
    scale = 2.0
    # Add 2 hidden neurons
    connections = [
        (H1, A), (H1, B), 
        (H2, A), (H2, B),
        (Y, H1), (Y, H2),
        (Y, A), (Y, B)
    ]
    
    for target, source in connections:
        # Initialize random matrix
        mat = [[random.gauss(0, scale) for _ in range(R)] for _ in range(R)]
        best_net.set_interaction_weights(target, source, mat)
        
    # Bias for H1, H2 and Y
    for cell in [H1, H2, Y]:
        best_net.set_local_weights(cell, [random.gauss(0, scale) for _ in range(R)])
    
    # Initial Eval
    best_mse, best_acc = evaluate_analog(best_net, inputs, targets, [A, B], Y)
    print(f"Initial: MSE={best_mse:.4f}, Acc={best_acc*100:.0f}%")
    
    epochs = 2000
    noise_scale = 1.0
    lr = 1.0 # Adaptive mutation rate
    
    for i in range(epochs):
        candidate_net = copy.deepcopy(best_net)
        
        # Mutate Connections
        for target, source in connections:
            mat = candidate_net.Theta[(target, source)]
            add_noise_to_matrix(mat, noise_scale * lr)
            
        # Mutate Biases
        for cell in [H1, H2, Y]:
            add_noise_to_vector(candidate_net.theta[cell], noise_scale * lr)
        
        # Evaluate
        mse, acc = evaluate_analog(candidate_net, inputs, targets, [A, B], Y)
        
        # Acceptance Criteria: Prefer Accuracy, then MSE
        improved = False
        if acc > best_acc:
            improved = True
        elif acc == best_acc and mse < best_mse:
            improved = True
            
        if improved:
            best_net = candidate_net
            best_mse = mse
            best_acc = acc
            print(f"Epoch {i}: Acc={best_acc*100:.0f}%, MSE={best_mse:.4f}")
            
            # Anneal learning rate
            if best_acc == 1.0 and best_mse < 0.1:
                lr = 0.5
            if best_mse < 0.05:
                print("Converged!")
                break
    
    # Final Verification
    print("\nFinal Verification (XOR):")
    for input_vals, target_val in zip(inputs, targets):
        # Run single inference
        best_net.reset_local_weights()
        for idx, val in zip([A, B], input_vals):
            bias = [-5.0] * R
            bias[val] = 10.0
            best_net.set_local_weights(idx, bias)
            
        best_net.state = [[0.5]*R for _ in range(num_cells)] # Neutral start
        best_net.run_until_stable(max_steps=30, temperature=0.3, noise=0.0)
        
        out_prob = best_net.state[Y][1] # Prob of being 1
        prediction = 1 if out_prob > 0.5 else 0
        
        status = "PASS" if prediction == target_val else "FAIL"
        print(f"XOR({input_vals[0]}, {input_vals[1]}) -> Prob(1)={out_prob:.3f} -> {prediction} | {status}")

if __name__ == "__main__":
    train_analog_xor()
