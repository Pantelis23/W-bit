import argparse
import sys
import os
import random
import csv
import time
import statistics
import hashlib

# Add parent directory to path to import wbit
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit.analog_network import AnalogWbitNetwork
from wbit.step_utils import summarize_steps

def compute_adaptive_n(grid, density, sigma, optimal_dist, adaptive_max_n, fallback_R):
    score = 0
    if grid is not None and grid >= 30:
        score += 1
    if density is not None and density >= 0.30:
        score += 1
    if sigma is not None and sigma >= 0.5:
        score += 1
    if optimal_dist is not None and grid is not None and optimal_dist >= grid:
        score += 1
    if score <= 0:
        n = 2
    elif score == 1:
        n = 2
    elif score == 2:
        n = 3
    else:
        n = 4
    max_n = adaptive_max_n if adaptive_max_n is not None else max(1, (fallback_R - 1) // 2)
    n = max(1, min(n, max_n))
    return n

def generate_obstacles(layout, density, W, H, target_x, target_y, rng=None):
    if rng is None:
        rng = random
        
    obstacles = set()
    num_obstacles = int(W * H * density)
    
    if layout == 'random':
        while len(obstacles) < num_obstacles:
            ox = rng.randint(0, W-1)
            oy = rng.randint(0, H-1)
            if (ox, oy) != (0, 0) and (ox, oy) != (target_x, target_y):
                obstacles.add((ox, oy))
                
    elif layout == 'box_canyon':
        for x in range(W-2):
            obstacles.add((x, 2))
        for x in range(2, W):
            obstacles.add((x, 5))
            
    elif layout == 'bottleneck':
        mid_x = W // 2
        mid_y = H // 2
        for y in range(H):
            if y != mid_y:
                obstacles.add((mid_x, y))
    
    elif layout == 'wall_with_gap':
        mid_x = W // 2
        gap_y = rng.randint(0, H-1)
        for y in range(H):
            if y != gap_y:
                obstacles.add((mid_x, y))
                
    elif layout == 'double_bottleneck':
        x1 = W // 3
        x2 = 2 * W // 3
        gap1_y = H // 2
        gap2_y = H // 2 
        for y in range(H):
            if y != gap1_y: obstacles.add((x1, y))
            if y != gap2_y: obstacles.add((x2, y))

    elif layout == 'maze_corridor':
        for y in range(1, H-1, 2):
            if (y // 2) % 2 == 0:
                 for x in range(W-1):
                     if (x, y) != (target_x, target_y) and (x, y) != (0,0):
                        obstacles.add((x, y))
            else:
                 for x in range(1, W):
                     if (x, y) != (target_x, target_y) and (x, y) != (0,0):
                        obstacles.add((x, y))
                        
    return obstacles

def stable_map_hash(layout, W, H, target_x, target_y, obstacles):
    sorted_obs = sorted(list(obstacles))
    payload = f"{layout}|{W}x{H}|{target_x},{target_y}|{sorted_obs}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]

def run_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'results.csv')
    summary_path = os.path.join(args.output_dir, 'summary.csv')
    write_header = not os.path.exists(csv_path)
    write_summary_header = not os.path.exists(summary_path)
    
    with open(csv_path, 'a', newline='') as f, open(summary_path, 'a', newline='') as f_sum:
        writer = csv.writer(f)
        sum_writer = csv.writer(f_sum)
        
        header = [
            'trial', 'base_seed', 'map_seed', 'noise_seed', 'layout', 'obstacle_density', 'sigma', 
            'grid', 'R', 'T', 'success', 'feasible', 'path_exists', 'penalty_used', 
            'steps', 'final_confidence', 'rcp', 'optimal_dist', 'active_cells', 
            'path_len', 'detour_ratio', 'n_obstacles', 'map_hash', 'mode', 'mode_variant', 'n_effective', 'R_effective', 'sat_count'
        ]
        
        summary_header = [
            'layout', 'grid', 'R', 'T', 'obstacle_density', 'sigma', 'trials',
            'feasible_rate', 'conditional_success', 'avg_rcp_success',
            'mean_detour_success', 'mean_path_success', 'avg_steps_success', 'avg_conf_success', 'avg_sat_count_success'
        ]

        def write_row(row):
            assert len(row) == len(header), f"Row/Header mismatch: {len(row)} vs {len(header)}"
            writer.writerow(row)

        def write_summary_row(row):
            assert len(row) == len(summary_header), f"Summary Row/Header mismatch: {len(row)} vs {len(summary_header)}"
            sum_writer.writerow(row)
        
        if write_header:
            writer.writerow(header)
        if write_summary_header:
            sum_writer.writerow(summary_header)
            
        # Global seed for loop control only (not for experiment logic)
        random.seed(args.seed)
        
        if args.layout != 'random':
            densities = [args.obstacle_density[0] if args.obstacle_density else 0.0]
        else:
            densities = args.obstacle_density if args.obstacle_density is not None else [0.1, 0.2, 0.3]
        
        if args.sigma_sweep:
            if args.sigma_list:
                sigmas = args.sigma_list
            else:
                sigmas = [0.0, 0.1, 0.2, 0.3, 0.5]
        else:
            sigmas = [args.sigma] if args.sigma is not None else [0.1]

        print(f"Starting Experiment A: Router Sweep")
        print(f"Layout: {args.layout}")
        print(f"Densities: {densities}")
        print(f"Sigmas: {sigmas}")
        print(f"Trials per config: {args.trials}")
        print(f"Base Seed: {args.seed}")
        
        for density in densities:
            invariance_cache = {} if (args.assert_map_invariance and args.sigma_sweep) else None
            for sigma in sigmas:
                feasible_count = 0
                success_count = 0
                rcp_success_sum = 0.0
                sat_count_success_sum = 0
                
                success_detour_ratios = []
                success_path_lens = []
                success_steps = []
                success_conf = []
                
                for trial in range(args.trials):
                    # Fix A: Decouple map seed from noise
                    map_seed = args.seed + trial + int(density * 1000)
                    noise_seed = map_seed + int(sigma * 10000)
                    
                    # Create dedicated RNG for map gen
                    rng_map = random.Random(map_seed)
                    
                    # Setup Grid
                    W = args.grid
                    H = args.grid
                    N = W * H
                    mode_variant = 'wbit'
                    if args.mode == 'binary':
                        mode_variant = 'binary_strict_R2' if getattr(args, 'binary_force_R2', False) else 'binary_quantized'
                        R_effective = 2 if getattr(args, 'binary_force_R2', False) else args.R
                        n_effective = max(1, (R_effective - 1) // 2)
                    elif args.mode == 'adaptive':
                        n_effective = compute_adaptive_n(args.grid, density, sigma, None, getattr(args, 'adaptive_max_n', None), args.R)
                        R_effective = max(3, 2 * n_effective + 1)
                        mode_variant = f"adaptive_heuristic_n{n_effective}"
                    else:
                        R_effective = args.R
                        n_effective = max(1, (R_effective - 1) // 2)
                    
                    net = AnalogWbitNetwork(N, R_effective, mode=args.mode)
                    
                    # Target
                    target_x, target_y = W - 1, H - 1
                    
                    # Obstacles (Using rng_map)
                    obstacles = generate_obstacles(args.layout, density, W, H, target_x, target_y, rng=rng_map)
                    n_obstacles = len(obstacles)
                    
                    # Map Hash (Stable)
                    map_hash = stable_map_hash(args.layout, W, H, target_x, target_y, obstacles)
                            
                    # BFS for Potential Field
                    grid_dist = [[999] * W for _ in range(H)]
                    queue = [(target_x, target_y, 0)]
                    grid_dist[target_y][target_x] = 0
                    
                    # Path existence check
                    path_exists = False
                    
                    visited_bfs = set([(target_x, target_y)])
                    head = 0
                    optimal_dist = 0
                    
                    while head < len(queue):
                        cx, cy, d = queue[head]
                        head += 1
                        if (cx, cy) == (0, 0):
                            path_exists = True
                            optimal_dist = d
                        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nx, ny = cx+dx, cy+dy
                            if 0 <= nx < W and 0 <= ny < H:
                                if (nx, ny) not in obstacles and grid_dist[ny][nx] == 999:
                                    grid_dist[ny][nx] = d + 1
                                    queue.append((nx, ny, d+1))
                                    visited_bfs.add((nx, ny))
                    
                    if not path_exists:
                        if args.layout == 'random':
                             for _ in range(10):
                                obstacles = generate_obstacles(args.layout, density, W, H, target_x, target_y, rng=rng_map)
                                n_obstacles = len(obstacles)
                                # Re-run BFS
                                grid_dist = [[999] * W for _ in range(H)]
                                queue = [(target_x, target_y, 0)]
                                grid_dist[target_y][target_x] = 0
                                head = 0
                                path_found_retry = False
                                while head < len(queue):
                                    cx, cy, d = queue[head]
                                    head += 1
                                    if (cx, cy) == (0, 0):
                                        path_found_retry = True
                                        optimal_dist = d
                                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                                        nx, ny = cx+dx, cy+dy
                                        if 0 <= nx < W and 0 <= ny < H:
                                            if (nx, ny) not in obstacles and grid_dist[ny][nx] == 999:
                                                grid_dist[ny][nx] = d + 1
                                                queue.append((nx, ny, d+1))
                                if path_found_retry:
                                    path_exists = True
                                    # Update hash if regenerated
                                    map_hash = stable_map_hash(args.layout, W, H, target_x, target_y, obstacles)
                                    break

                    if invariance_cache is not None:
                        prev_hash = invariance_cache.get(trial)
                        if prev_hash is None:
                            invariance_cache[trial] = map_hash
                        else:
                            assert prev_hash == map_hash, (
                                f"Map hash changed for trial {trial} at sigma {sigma}: "
                                f"{prev_hash} vs {map_hash}"
                            )

                    if not path_exists:
                        penalty_rcp = (N * args.max_steps * (R_effective * R_effective)) / 1.0
                        write_row([
                            trial, args.seed, map_seed, noise_seed, args.layout, density, sigma,
                            args.grid, R_effective, args.T, 0, 0, 0, 1, 
                            0, 0.0, penalty_rcp, 0, 0, 0, 0.0, n_obstacles, map_hash, args.mode, mode_variant, n_effective, R_effective, 0
                        ])
                        continue
                    
                    feasible_count += 1

                    # Configure Weights
                    for y in range(H):
                        for x in range(W):
                            cell_id = y * W + x
                            if (x, y) == (target_x, target_y): continue
                            if (x, y) in obstacles:
                                bias = [-5.0] * R_effective
                                bias[0] = 5.0 
                                net.set_local_weights(cell_id, bias)
                                continue
                            current_dist = grid_dist[y][x]
                            if current_dist == 999: continue
                            bias = [0.0] * R_effective
                            def set_dir(idx):
                                if idx < len(bias):
                                    bias[idx] = 5.0
                            if y > 0 and grid_dist[y-1][x] < current_dist: set_dir(1)
                            if x < W-1 and grid_dist[y][x+1] < current_dist: set_dir(2)
                            if y < H-1 and grid_dist[y+1][x] < current_dist: set_dir(3)
                            if x > 0 and grid_dist[y][x-1] < current_dist: set_dir(4)
                            net.set_local_weights(cell_id, bias)

                    # Run Simulation (Using noise_seed)
                    # Seed global random for noise (net uses random.gauss)
                    random.seed(noise_seed)
                    steps_converged = net.run_until_stable(max_steps=args.max_steps, temperature=args.T, noise=sigma)
                    
                    # Capture AET Stats
                    aet_stats = net.get_aet_stats()
                    sat_count = aet_stats.get('sat_count', 0)
                    
                    cx, cy = 0, 0
                    path_steps = 0
                    max_path_search = W * H
                    reached = False
                    hard_states = net.get_hard_state()
                    
                    while path_steps < max_path_search:
                        if (cx, cy) == (target_x, target_y):
                            reached = True
                            break
                        cell_id = cy * W + cx
                        direction = hard_states[cell_id]
                        if direction == 1: cy -= 1
                        elif direction == 2: cx += 1
                        elif direction == 3: cy += 1
                        elif direction == 4: cx -= 1
                        else: break
                        if not (0 <= cx < W and 0 <= cy < H): break
                        path_steps += 1
                    
                    steps_taken_raw, reason = summarize_steps(reached, steps_converged, args.max_steps, path_exists=path_exists, converged=(steps_converged < args.max_steps))
                    steps_taken = steps_taken_raw
                    if reached:
                        steps_taken = max(steps_taken, path_steps)
                        if args.mode == 'binary':
                            steps_taken = max(steps_taken, 1)
                    n_active = path_steps if reached else N 
                    rcp = net.calculate_RCP(steps_taken, i_out=1.0, n_active_cells=n_active)
                    if args.mode == 'binary' and rcp == 0:
                        # Avoid zero RCP in binary baseline
                        rcp = net.calculate_RCP(max(steps_taken, 1), i_out=1.0, n_active_cells=max(n_active, 1))
                    start_state = net.state[0]
                    final_conf = max(start_state)
                    path_len = path_steps if reached else 0
                    detour_ratio = (path_len / optimal_dist) if reached and optimal_dist > 0 else 0.0
                    
                    if args.debug_map_hash:
                        print(f"Trial {trial}, Sigma {sigma}: MapHash {map_hash}")

                    write_row([
                        trial, args.seed, map_seed, noise_seed, args.layout, density, sigma,
                        args.grid, R_effective, args.T, 1 if reached else 0, 1, 1, 0, 
                        steps_taken, final_conf, rcp, optimal_dist, n_active, 
                        path_len, detour_ratio, n_obstacles, map_hash, args.mode, mode_variant, n_effective, R_effective, sat_count
                    ])

                    if getattr(args, 'debug_metrics', False):
                        print({
                            "mode": args.mode,
                            "layout": args.layout,
                            "grid": args.grid,
                            "density": density,
                            "sigma": sigma,
                            "success": 1 if reached else 0,
                            "steps": steps_taken,
                            "rcp": rcp,
                            "reason": reason,
                            "sat_count": sat_count
                        })
                    
                    if reached:
                        success_count += 1
                        rcp_success_sum += rcp
                        sat_count_success_sum += sat_count
                        success_path_lens.append(path_len)
                        success_steps.append(steps_taken)
                        success_conf.append(final_conf)
                        if optimal_dist > 0:
                            success_detour_ratios.append(detour_ratio)
                # Reporting
                feasible_rate = feasible_count / args.trials
                conditional_success = (success_count / feasible_count) if feasible_count > 0 else 0.0
                avg_rcp_success = (rcp_success_sum / success_count) if success_count > 0 else 0.0
                avg_sat_count_success = (sat_count_success_sum / success_count) if success_count > 0 else 0.0
                
                mean_detour = statistics.mean(success_detour_ratios) if success_detour_ratios else 0.0
                mean_path = statistics.mean(success_path_lens) if success_path_lens else 0.0
                
                avg_steps_success = statistics.mean(success_steps) if success_steps else 0.0
                avg_final_conf_success = statistics.mean(success_conf) if success_conf else 0.0
                
                write_summary_row([
                    args.layout, args.grid, R_effective, args.T, density, sigma, args.trials,
                    feasible_rate, conditional_success, avg_rcp_success,
                    mean_detour, mean_path, avg_steps_success, avg_final_conf_success, avg_sat_count_success
                ])
                
                print(f"Layout {args.layout}, Density {density}, Sigma {sigma}:")
                print(f"  Feasible: {feasible_count}/{args.trials}")
                print(f"  Conditional Success: {conditional_success:.2f}")
                print(f"  Avg RCP (success-only): {avg_rcp_success:.1f}")
                print(f"  Detour Ratio (success): Mean={mean_detour:.2f}")
                print(f"  Avg Steps (success): {avg_steps_success:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--grid', type=int, default=10)
    parser.add_argument('--R', type=int, default=5)
    parser.add_argument('--T', type=float, default=0.2)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--obstacle_density', type=float, nargs='+', help='List of densities')
    parser.add_argument('--layout', type=str, default='random', choices=['random', 'box_canyon', 'bottleneck', 'wall_with_gap', 'double_bottleneck', 'maze_corridor'], help='Obstacle layout type')
    parser.add_argument('--output_dir', type=str, default='results/expA')
    
    parser.add_argument('--sigma_sweep', action='store_true', help="Enable sigma sweep per density")
    parser.add_argument('--sigma_list', type=float, nargs='+', help='List of sigmas to sweep', default=None)
    parser.add_argument('--debug_map_hash', action='store_true', help="Print map hash for validation")
    parser.add_argument('--assert_map_invariance', action='store_true', help="Assert map stability across sigmas")
    parser.add_argument('--mode', type=str, default='wbit', choices=['wbit', 'binary', 'adaptive'], help='Logic substrate mode (wbit default, binary placeholder)')
    parser.add_argument('--debug_metrics', action='store_true', help='Print per-trial metrics for debugging')
    parser.add_argument('--binary_force_R2', action='store_true', help='Force R=2 in binary mode (strict ablation)')
    parser.add_argument('--adaptive_max_n', type=int, default=None, help='Max n (R=2n+1) for adaptive mode')
    
    args = parser.parse_args()
    run_experiment(args)