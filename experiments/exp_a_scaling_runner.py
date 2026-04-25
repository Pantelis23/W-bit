import argparse
import subprocess
import os
import sys
import csv
import statistics

def run_scaling_sweep(args):
    print("=== Experiment A Scaling Runner ===")
    print(f"Grids: {args.grids}")
    print(f"Densities: {args.densities}")
    print(f"Layout: {args.layout}")
    
    python_exe = sys.executable
    exp_script = os.path.join(os.path.dirname(__file__), 'exp_a_router_sweep.py')
    
    for grid in args.grids:
        print(f"\n--- Running Grid {grid}x{grid} ---")
        
        out_dir = os.path.join(args.output_dir, f"grid_{grid}")
        
        cmd = [
            python_exe,
            exp_script,
            '--layout', args.layout,
            '--trials', str(args.trials),
            '--seed', str(args.seed),
            '--grid', str(grid),
            '--output_dir', out_dir
        ]
        
        # Add densities
        cmd.append('--obstacle_density')
        cmd.extend([str(d) for d in args.densities])
        
        subprocess.run(cmd, check=True)
        
        # A3: Parse and print summary
        results_file = os.path.join(out_dir, 'results.csv')
        if os.path.exists(results_file):
             with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Group by density
                by_density = {}
                for r in rows:
                    d = float(r['obstacle_density'])
                    if d not in by_density:
                        by_density[d] = []
                    by_density[d].append(r)
                
                print(f"\nSummary for Grid {grid}:")
                print(f"{'Density':<8} {'Feasible':<10} {'CondSuccess':<12} {'Avg RCP':<10} {'MeanDetour':<12} {'MeanPath':<10}")
                print("-" * 70)
                
                for d in sorted(by_density.keys()):
                    trials = by_density[d]
                    feasible_count = sum(1 for r in trials if int(r.get('feasible', r.get('path_exists', 0))) == 1)
                    success_count = sum(1 for r in trials if int(r['success']) == 1)
                    
                    # success-only metrics
                    rcps = [float(r['rcp']) for r in trials if int(r['success']) == 1]
                    detours = [float(r['detour_ratio']) for r in trials if int(r['success']) == 1]
                    paths = [float(r['path_len']) for r in trials if int(r['success']) == 1]
                    
                    feasible_rate = feasible_count / len(trials)
                    cond_success = (success_count / feasible_count) if feasible_count > 0 else 0.0
                    avg_rcp = statistics.mean(rcps) if rcps else 0.0
                    mean_detour = statistics.mean(detours) if detours else 0.0
                    mean_path = statistics.mean(paths) if paths else 0.0
                    
                    print(f"{d:<8.2f} {feasible_rate:<10.2f} {cond_success:<12.2f} {avg_rcp:<10.1f} {mean_detour:<12.2f} {mean_path:<10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grids', type=int, nargs='+', default=[10, 20, 30])
    parser.add_argument('--densities', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layout', type=str, default='random')
    parser.add_argument('--output_dir', type=str, default='results/expA_scaling')
    
    args = parser.parse_args()
    run_scaling_sweep(args)
