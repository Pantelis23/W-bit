import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from wbit.level3_adaptive_network import Level3WbitNetwork

def run_adaptive_logic_test():
    print("=== LEVEL 3 W-BIT TEST: Adaptive Logic / Variable Reffective ===")
    
    # Create a fabric capable of 9 analog states (Dense Matrix mode)
    fabric = Level3WbitNetwork(num_cells=2, R_max=9, mode="adaptive")
    
    # We want Cell 0 to push Cell 1 into the exact same state.
    # A simple "copy" compatibility matrix.
    R = fabric.R_max
    copy_matrix = [[0.0 for _ in range(R)] for _ in range(R)]
    for i in range(R):
        copy_matrix[i][i] = 5.0 # Strong diagonal (preference to match)
        
    fabric.set_interaction_weights(1, 0, copy_matrix)
    
    # Force Cell 0 into State 8 (An edge precision state)
    fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R)]
    
    print("\n[Scenario 1: Clean Environment, High Complexity Task]")
    # The OS detects low noise. It should allow full R=9 precision.
    fabric.schedule_capacity(task_complexity=0.9, environmental_noise=0.0)
    
    # Let the network settle
    for _ in range(10):
        fabric.step(temperature=0.1, dt=0.5)
        # Keep Cell 0 clamped
        fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R)]
        
    # Check Cell 1's state
    cell1_hard = max(range(R), key=lambda idx: fabric.state[1][idx])
    print(f"Cell 1 settled into State: {cell1_hard} (Expected: 8)")
    
    
    print("\n[Scenario 2: High Noise Environment]")
    # The OS detects a massive heat/noise spike. 
    # To prevent catastrophic routing failures, it clamps the fabric to Phi-mode (R=3).
    fabric.schedule_capacity(task_complexity=0.9, environmental_noise=0.8)
    
    # Keep Cell 0 trying to broadcast State 8, but the hardware comparators are now wide.
    fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R)]
    
    for _ in range(10):
        fabric.step(temperature=0.1, dt=0.5)
        fabric.state[0] = [1.0 if r == 8 else 0.0 for r in range(R)]
        
    cell1_probs = [round(p, 2) for p in fabric.state[1]]
    cell1_hard = max(range(R), key=lambda idx: fabric.state[1][idx])
    
    print(f"Cell 1 settled into State: {cell1_hard}")
    print(f"Cell 1 Probability Vector: {cell1_probs}")
    print("Notice how the probability mass is binned to the centers of the 3 wide energy wells.")

if __name__ == "__main__":
    run_adaptive_logic_test()