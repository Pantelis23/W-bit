import sys
import os
import copy
import random

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit import WDitNetwork

def evaluate(net, inputs, targets, input_indices, output_index):
    correct = 0
    R = net.R
    
    for input_vals, target_val in zip(inputs, targets):
        # Clamp inputs
        net.reset_local_weights()
        for idx, val in zip(input_indices, input_vals):
            bias = [0.0] * R
            bias[val] = 100.0
            net.set_local_weights(idx, bias)
            
        # Reset output to random
        net.state[output_index] = random.randint(0, R-1)
        
        # Run
        net.run_until_stable(max_steps=20)
        
        if net.state[output_index] == target_val:
            correct += 1
            
    return correct / len(inputs)

def add_noise_to_matrix(matrix, scale):
    """Mutates a matrix (list of lists) in place."""
    for r in range(len(matrix)):
        for c in range(len(matrix[r])):
            matrix[r][c] += random.gauss(0, scale)

def add_noise_to_vector(vector, scale):
    for i in range(len(vector)):
        vector[i] += random.gauss(0, scale)

def train_max_gate():
    print("=== W-bit Logic Training Demo (Pure Python) ===")
    print("Objective: Learn Y = MAX(A, B) via random search.")
    
    R = 3
    A, B, Y = 0, 1, 2
    
    # Generate Truth Table
    inputs = []
    targets = []
    for a in range(R):
        for b in range(R):
            inputs.append((a, b))
            targets.append(max(a, b))
            
    # Initialize Network
    best_net = WDitNetwork(3, R)
    # Initialize zero interactions for A->Y and B->Y
    best_net.set_interaction_weights(Y, A, [[0.0]*R for _ in range(R)])
    best_net.set_interaction_weights(Y, B, [[0.0]*R for _ in range(R)])
    
    best_acc = evaluate(best_net, inputs, targets, [A, B], Y)
    print(f"Initial Accuracy: {best_acc * 100:.1f}%")
    
    epochs = 2000
    noise_scale = 0.5
    
    for i in range(epochs):
        candidate_net = copy.deepcopy(best_net)
        
        # Mutate A->Y
        mat_ay = candidate_net.Theta[(Y, A)]
        add_noise_to_matrix(mat_ay, noise_scale)
        
        # Mutate B->Y
        mat_by = candidate_net.Theta[(Y, B)]
        add_noise_to_matrix(mat_by, noise_scale)
        
        # Mutate Y bias
        add_noise_to_vector(candidate_net.theta[Y], noise_scale)
        
        # Evaluate
        acc = evaluate(candidate_net, inputs, targets, [A, B], Y)
        
        if acc > best_acc:
            best_acc = acc
            best_net = candidate_net
            print(f"Epoch {i}: New Best Accuracy = {best_acc * 100:.1f}%")
            if best_acc == 1.0:
                print("Converged!")
                break
                
    # Final Test
    print("\nFinal Verification:")
    for input_vals, target_val in zip(inputs, targets):
        best_net.reset_local_weights()
        for idx, val in zip([A, B], input_vals):
            bias = [0.0] * R
            bias[val] = 100.0
            best_net.set_local_weights(idx, bias)
        best_net.state[Y] = random.randint(0, R-1)
        best_net.run_until_stable()
        
        out = best_net.state[Y]
        status = "PASS" if out == target_val else "FAIL"
        print(f"MAX({input_vals[0]}, {input_vals[1]}) = {out} (Expected {target_val}) | {status}")

if __name__ == "__main__":
    train_max_gate()