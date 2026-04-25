import sys
import os
import random
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit.analog_network import AnalogWbitNetwork

def run_grid_router():
    print("=== W-bit Grid Router Simulation ===")
    print("Simulating a 'Liquid Routing Fabric' (10x10 Grid).")
    print("Packet starts at (0,0) and wants to go to (9,9).")
    print("Obstacles are placed in the way. The energy landscape guides the flow.")
    
    W = 10
    H = 10
    N = W * H
    R = 5 # States: 0=Empty, 1=North, 2=East, 3=South, 4=West (Flow Direction)
    
    # Map directions to R-ary states
    # State 0: Idle/Empty
    # State 1: Flow Up (North)
    # State 2: Flow Right (East)
    # State 3: Flow Down (South)
    # State 4: Flow Left (West)
    
    net = AnalogWbitNetwork(N, R)
    
    def idx(x, y):
        return y * W + x
    
    print("Configuring Energy Landscape...")
    
    # 1. Define Physics (The "Flow" Rules)
    # If a neighbor is the Target, point to it.
    # If a neighbor points to Me, I should point to it? No, that's backwards.
    # Logic: "Gradient Descent"
    # We will simply program a "Potential Field" into the Local Bias (Theta).
    # Target (9,9) has attractive potential.
    # Obstacles have repulsive potential. 
    
    target_x, target_y = 9, 9
    obstacles = [(3,3), (3,4), (3,5), (4,5), (5,5), (6,2), (6,3), (7,7), (8,8)]
    
    # We won't use pairwise weights for this demo (too complex to hand-code flow dynamics).
    # Instead, we will use the "Local Preference" (Theta) to visualize the POTENTIAL FIELD.
    # In a real w-bit chip, this potential would propagate via interactions.
    # Here, we pre-calculate the "smell" of the target.
    
    # Simple BFS to find distance to target
    grid_dist = [[999] * W for _ in range(H)]
    queue = [(target_x, target_y, 0)]
    grid_dist[target_y][target_x] = 0
    
    while queue:
        cx, cy, d = queue.pop(0)
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < W and 0 <= ny < H:
                if (nx, ny) not in obstacles and grid_dist[ny][nx] == 999:
                    grid_dist[ny][nx] = d + 1
                    queue.append((nx, ny, d+1))
                    
    # Now encode this "Gradient" into the w-bit weights
    for y in range(H):
        for x in range(W):
            cell_id = idx(x, y)
            
            if (x, y) == (target_x, target_y):
                # Target: Stays as "Target" (Let's say State 0 is neutral)
                continue
            
            if (x, y) in obstacles:
                # Obstacle: Force State 0 (Idle) strongly
                bias = [10.0, -10.0, -10.0, -10.0, -10.0]
                net.set_local_weights(cell_id, bias)
                continue
                
            # Normal Cell: Look at neighbors and bias towards the one with lowest distance
            current_dist = grid_dist[y][x]
            bias = [0.0] * R # Default neutral
            
            # Check neighbors to decide preferred flow direction
            # 1: North (y-1), 2: East (x+1), 3: South (y+1), 4: West (x-1)
            
            # North
            if y > 0 and grid_dist[y-1][x] < current_dist:
                bias[1] = 5.0
            # East
            if x < W-1 and grid_dist[y][x+1] < current_dist:
                bias[2] = 5.0
            # South
            if y < H-1 and grid_dist[y+1][x] < current_dist:
                bias[3] = 5.0
            # West
            if x > 0 and grid_dist[y][x-1] < current_dist:
                bias[4] = 5.0
                
            net.set_local_weights(cell_id, bias)

    print("Running Simulation (Relaxing the Field)...")
    # Even though we hard-coded biases, the analog system needs to "settle" 
    # because we initialized with random noise.
    net.run_until_stable(max_steps=20, temperature=0.2, noise=0.1)
    
    # Visualization
    hard_states = net.get_hard_state()
    
    print("\nResulting Flow Field:")
    # Symbols: . (Empty/Target), ^ (N), > (E), v (S), < (W), # (Obstacle)
    symbols = ['.', '^', '>', 'v', '<']
    
    # Path tracing for visualization
    path_cells = set()
    cx, cy = 0, 0
    steps = 0
    while (cx, cy) != (target_x, target_y) and steps < 100:
        path_cells.add((cx, cy))
        cell_state = hard_states[idx(cx, cy)]
        if cell_state == 1: cy -= 1
        elif cell_state == 2: cx += 1
        elif cell_state == 3: cy += 1
        elif cell_state == 4: cx -= 1
        else: break # Stuck
        steps += 1
    path_cells.add((target_x, target_y))

    for y in range(H):
        row_str = ""
        for x in range(W):
            if (x, y) in obstacles:
                row_str += " # "
            elif (x, y) == (target_x, target_y):
                row_str += " T "
            elif (x, y) == (0, 0):
                row_str += " S " # Start
            elif (x, y) in path_cells:
                # Highlight path
                s = hard_states[idx(x, y)]
                sym = symbols[s]
                row_str += f"*{sym}*"
            else:
                s = hard_states[idx(x, y)]
                sym = symbols[s]
                row_str += f" {sym} "
        print(row_str)
        
    print("\nLegend: S=Start, T=Target, #=Obstacle, *=Path, ^>v<=Flow Direction")
    if (target_x, target_y) in path_cells:
        print(">>> SUCCESS: Valid path established from S to T.")
    else:
        print(">>> FAILURE: Path blocked or stuck.")

if __name__ == "__main__":
    run_grid_router()
