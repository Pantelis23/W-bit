import argparse
import subprocess
import os
import sys

def run_phase1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expA_out', type=str, default='results/expA')
    parser.add_argument('--expB_out', type=str, default='results/expB')
    parser.add_argument('--expB_grid_out', type=str, default='results/expB_grid')
    parser.add_argument('--expC_out', type=str, default='results/expC')
    parser.add_argument('--report_out', type=str, default='results/phase1_report.csv')
    parser.add_argument('--base_results_dir', type=str, default=None, help='Optional base dir to derive all outputs')
    parser.add_argument('--smoke', action='store_true', help='Run fast smoke settings')
    parser.add_argument('--run_expB_grid', action='store_true', help='Also run Exp B grid sweep')
    args = parser.parse_args()

    if args.base_results_dir:
        base = os.path.abspath(args.base_results_dir)
        args.expA_out = os.path.join(base, 'expA')
        args.expB_out = os.path.join(base, 'expB')
        args.expB_grid_out = os.path.join(base, 'expB_grid')
        args.expC_out = os.path.join(base, 'expC')
        args.report_out = os.path.join(base, 'phase1_report.csv')

    print("=== w-dit Phase 1 Validation Runner ===")
    
    python_exe = sys.executable
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(base_dir, 'experiments')
    analysis_dir = os.path.join(base_dir, 'analysis')
    
    # Smoke overrides
    expA_trials = '100'
    expA_sigma = '0.1'
    expA_grid = '10'
    expA_R = '5'
    expA_density_args = []

    expB_trials = '100'
    expB_R = '3'

    expC_trials = '10'
    expC_R = '2'
    expC_population = None
    expC_elite_k = None
    expC_restarts = None
    expC_max_epochs = None

    expB_grid_trials = '50'

    if args.smoke:
        expA_trials = '5'
        expA_sigma = '0.1'
        expA_grid = '20'
        expA_R = '5'
        expA_density_args = ['--obstacle_density', '0.38']

        expB_trials = '5'

        expC_trials = '3'
        expC_population = '2'
        expC_elite_k = '1'
        expC_restarts = '1'
        expC_max_epochs = '300'

        expB_grid_trials = '5'

    # Experiment A
    print("\nRunning Experiment A: Router Sweep...")
    cmd_a = [
        python_exe, os.path.join(exp_dir, 'exp_a_router_sweep.py'),
        '--trials', expA_trials,
        '--sigma', expA_sigma,
        '--grid', expA_grid,
        '--R', expA_R,
        '--output_dir', args.expA_out
    ]
    if expA_density_args:
        cmd_a.extend(expA_density_args)
    subprocess.run(cmd_a, check=True, cwd=base_dir)
    
    # Experiment B
    print("\nRunning Experiment B: Noise Breakdown...")
    cmd_b = [
        python_exe, os.path.join(exp_dir, 'exp_b_noise_breakdown.py'),
        '--trials', expB_trials,
        '--R', expB_R,
        '--output_dir', args.expB_out
    ]
    subprocess.run(cmd_b, check=True, cwd=base_dir)

    if args.run_expB_grid:
        print("\nRunning Experiment B Grid Sweep...")
        cmd_bg = [
            python_exe, os.path.join(exp_dir, 'exp_b_weight_noise_grid.py'),
            '--trials', expB_grid_trials,
            '--output_dir', args.expB_grid_out
        ]
        subprocess.run(cmd_bg, check=True, cwd=base_dir)
    
    # Experiment C
    print("\nRunning Experiment C: Learning Search...")
    cmd_c = [
        python_exe, os.path.join(exp_dir, 'exp_c_learning_search.py'),
        '--trials', expC_trials,
        '--R', expC_R,
        '--output_dir', args.expC_out
    ]
    if expC_population:
        cmd_c.extend(['--population', expC_population])
    if expC_elite_k:
        cmd_c.extend(['--elite_k', expC_elite_k])
    if expC_restarts:
        cmd_c.extend(['--restarts', expC_restarts])
    if expC_max_epochs:
        cmd_c.extend(['--max_epochs', expC_max_epochs])
    subprocess.run(cmd_c, check=True, cwd=base_dir)

    # Aggregate Phase 1 report
    print("\nAggregating Phase 1 report...")
    cmd_report = [
        python_exe, os.path.join(analysis_dir, 'aggregate_phase1_report.py'),
        '--expA_dir', args.expA_out,
        '--expB_dir', args.expB_grid_out,
        '--expB_noise_dir', args.expB_out,
        '--expC_dir', args.expC_out,
        '--out', args.report_out
    ]
    subprocess.run(cmd_report, check=True, cwd=base_dir)
    
    print("\n=== Phase 1 Complete ===")
    print("Results locations:")
    print(f"  Exp A: {args.expA_out}")
    print(f"  Exp B (noise): {args.expB_out}")
    print(f"  Exp B (grid, if run separately): {args.expB_grid_out}")
    print(f"  Exp C: {args.expC_out}")
    print(f"  Report: {args.report_out}")

if __name__ == "__main__":
    run_phase1()
