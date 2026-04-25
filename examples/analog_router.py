import sys
import os
import random

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit.analog_network import AnalogWbitNetwork

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def eye(n, scale=1.0):
    mat = zeros(n, n)
    for i in range(n):
        mat[i][i] = scale
    return mat

def run_analog_router_demo():
    print("=== Analog W-bit Router Demo (Noisy & Continuous) ===")
    print("Simulating a 'Memristor-like' settling process with noise.")
    
    # R=3 (0=Null, 1=DataA, 2=DataB)
    R = 3
    net = AnalogWbitNetwork(num_cells=4, R=R)
    
    C, D, O1, O2 = 0, 1, 2, 3
    
    # Weights (Higher values because Softmax needs drive to sharpen)
    ALPHA = 5.0   # Copy strength 
    BETA = 20.0   # Veto strength
    
    # --- D -> O1 (Copy) ---
    net.set_interaction_weights(O1, D, eye(R, ALPHA))
    
    # --- D -> O2 (Copy) ---
    net.set_interaction_weights(O2, D, eye(R, ALPHA))
    
    # --- C -> O1 (Gate) ---
    # If C=1 (index 1), Veto O1 (Target 0)
    w_gate_o1 = zeros(R, R)
    w_gate_o1[0][1] = BETA 
    net.set_interaction_weights(O1, C, w_gate_o1)
    
    # --- C -> O2 (Gate) ---
    # If C=0 (index 0), Veto O2 (Target 0)
    w_gate_o2 = zeros(R, R)
    w_gate_o2[0][0] = BETA
    net.set_interaction_weights(O2, C, w_gate_o2)
    
    test_cases = [
        (0, 1), # Route to O1
        (1, 2), # Route to O2
        (0, 2), # Route to O1
    ]
    
    print(f"\nSimulation Parameters:")
    print(f"Noise Level: 0.5 (Gaussian noise added to every cell every step)")
    print(f"Temperature: 0.3 (Low temp = sharp logic)")
    print(f"Time Step (dt): 0.2 (Simulating capacitance/delay)")
    print("-" * 60)
    
    for c_val, d_val in test_cases:
        # Reset local biases
        net.reset_local_weights()
        
        # Clamp inputs using strong biases (Voltage inputs)
        c_bias = [-5.0] * R # Default low voltage
        c_bias[c_val] = 10.0 # High voltage for selected state
        net.set_local_weights(C, c_bias)
        
        d_bias = [-5.0] * R
        d_bias[d_val] = 10.0
        net.set_local_weights(D, d_bias)
        
        # Reset state to neutral/random
        net.state = [[random.uniform(0.3, 0.35) for _ in range(R)] for _ in range(4)]
        
        # Run Simulation Step-by-Step
        print(f"\nCase: Control={c_val}, Data={d_val}")
        print("Steps | O1 (Null/A/B) Probabilities     | O2 (Null/A/B) Probabilities")
        
        for step in range(1, 21): # 20 steps
            net.step(temperature=0.3, noise_level=0.5, dt=0.2)
            
            if step % 4 == 0:
                s_o1 = net.state[O1]
                s_o2 = net.state[O2]
                
                # Format prob strings
                o1_str = f"[{s_o1[0]:.2f}, {s_o1[1]:.2f}, {s_o1[2]:.2f}]"
                o2_str = f"[{s_o2[0]:.2f}, {s_o2[1]:.2f}, {s_o2[2]:.2f}]"
                print(f"{step:4d}  | {o1_str} | {o2_str}")
        
        # Final Verification
        hard_res = net.get_hard_state()
        print(f"Final Discrete State: O1={hard_res[O1]}, O2={hard_res[O2]}")
        
        # Expected
        exp_o1 = d_val if c_val == 0 else 0
        exp_o2 = d_val if c_val == 1 else 0
        
        if hard_res[O1] == exp_o1 and hard_res[O2] == exp_o2:
            print(">> SUCCESS: Signal stabilized correctly despite noise.")
        else:
            print(">> FAILURE: Noise disrupted the logic.")

if __name__ == "__main__":
    run_analog_router_demo()
