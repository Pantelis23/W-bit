import sys
import os
import random

# Add parent directory to path to import wbit
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit import WDitNetwork

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def eye(n, scale=1.0):
    mat = zeros(n, n)
    for i in range(n):
        mat[i][i] = scale
    return mat

def run_router_demo():
    print("=== W-bit Hand-Coded Router Demo (Pure Python) ===")
    print("Objective: Build a 1-to-2 Router using weight-defined logic.")
    
    # R=3 (0=Null, 1=DataA, 2=DataB)
    R = 3
    net = WDitNetwork(num_cells=4, R=R)
    
    # Indices
    C, D, O1, O2 = 0, 1, 2, 3
    
    # Define Weights
    ALPHA = 2.0  # Copy strength (Data -> Output)
    BETA = 10.0  # Veto strength (Control -> Output)
    
    # --- Interaction D -> O1 (Copy) ---
    w_copy = eye(R, scale=ALPHA)
    net.set_interaction_weights(O1, D, w_copy)
    
    # --- Interaction D -> O2 (Copy) ---
    net.set_interaction_weights(O2, D, w_copy)
    
    # --- Interaction C -> O1 (Gate) ---
    # If C=1, Veto O1 (Force 0)
    # We want matrix[0][1] (Target 0, Source 1) to be high.
    w_gate_o1 = zeros(R, R)
    w_gate_o1[0][1] = BETA 
    net.set_interaction_weights(O1, C, w_gate_o1)
    
    # --- Interaction C -> O2 (Gate) ---
    # If C=0, Veto O2 (Force 0)
    # We want matrix[0][0] (Target 0, Source 0) to be high.
    w_gate_o2 = zeros(R, R)
    w_gate_o2[0][0] = BETA
    net.set_interaction_weights(O2, C, w_gate_o2)
    
    # Test Cases
    test_cases = [
        (0, 1), # C=0, D=1 -> Expect O1=1, O2=0
        (0, 2), # C=0, D=2 -> Expect O1=2, O2=0
        (1, 1), # C=1, D=1 -> Expect O1=0, O2=1
        (1, 2), # C=1, D=2 -> Expect O1=0, O2=2
        (0, 0), # C=0, D=0 -> Expect O1=0, O2=0
    ]
    
    print(f"\nNetwork Params: R={R}, Copy={ALPHA}, Veto={BETA}")
    print(f"{ 'Control':<8} { 'Data':<8} | { 'Out1':<5} { 'Out2':<5} | { 'Result':<10}")
    print("-" * 50)
    
    success_count = 0
    
    for c_val, d_val in test_cases:
        # 1. Clamp Inputs via strong Local Preferences (Theta)
        net.reset_local_weights()
        
        # Clamp C
        c_bias = [0.0] * R
        c_bias[c_val] = 50.0 
        net.set_local_weights(C, c_bias)
        
        # Clamp D
        d_bias = [0.0] * R
        d_bias[d_val] = 50.0
        net.set_local_weights(D, d_bias)
        
        # Initialize outputs to random
        net.state[O1] = random.randint(0, R-1)
        net.state[O2] = random.randint(0, R-1)
        
        # Run relaxation
        net.run_until_stable()
        
        # Check results
        o1_val = net.state[O1]
        o2_val = net.state[O2]
        
        # Verify logic
        expected_o1 = d_val if c_val == 0 else 0
        expected_o2 = d_val if c_val == 1 else 0
        
        is_correct = (o1_val == expected_o1) and (o2_val == expected_o2)
        if is_correct: success_count += 1
        
        status = "PASS" if is_correct else "FAIL"
        print(f"{c_val:<8} {d_val:<8} | {o1_val:<5} {o2_val:<5} | {status}")

    print("-" * 50)
    print(f"Accuracy: {success_count}/{len(test_cases)}")

if __name__ == "__main__":
    run_router_demo()
