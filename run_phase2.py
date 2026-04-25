import argparse
import os
import subprocess
import sys

def run_phase2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_results_dir', type=str, default=None, help='Optional base dir to derive all outputs')
    parser.add_argument('--smoke', action='store_true', help='Run fast smoke settings')
    parser.add_argument('--run_expB_grid', action='store_true', help='Also run Exp B grid sweep')

    parser.add_argument('--wbit_expA_out', type=str, default='results/phase2/wbit/expA')
    parser.add_argument('--wbit_expB_out', type=str, default='results/phase2/wbit/expB')
    parser.add_argument('--wbit_expB_grid_out', type=str, default='results/phase2/wbit/expB_grid')
    parser.add_argument('--wbit_expC_out', type=str, default='results/phase2/wbit/expC')

    parser.add_argument('--binary_expA_out', type=str, default='results/phase2/binary/expA')
    parser.add_argument('--binary_expB_out', type=str, default='results/phase2/binary/expB')
    parser.add_argument('--binary_expB_grid_out', type=str, default='results/phase2/binary/expB_grid')
    parser.add_argument('--binary_expC_out', type=str, default='results/phase2/binary/expC')
    parser.add_argument('--adaptive_expA_out', type=str, default='results/phase2/adaptive/expA')
    parser.add_argument('--adaptive_expB_out', type=str, default='results/phase2/adaptive/expB')
    parser.add_argument('--adaptive_expB_grid_out', type=str, default='results/phase2/adaptive/expB_grid')
    parser.add_argument('--adaptive_expC_out', type=str, default='results/phase2/adaptive/expC')

    parser.add_argument('--report_out', type=str, default='results/phase2/phase2_report.csv')
    parser.add_argument('--expC_population', type=int, default=None, help='Override Exp C population (non-smoke default=8)')
    parser.add_argument('--expC_elite_k', type=int, default=None, help='Override Exp C elite_k (non-smoke default=2)')
    parser.add_argument('--expC_restarts', type=int, default=None, help='Override Exp C restarts (non-smoke default=5)')
    parser.add_argument('--expC_trials', type=int, default=None, help='Override Exp C trials (non-smoke default=20)')
    parser.add_argument('--expC_max_epochs', type=int, default=None, help='Override Exp C max_epochs')
    parser.add_argument('--debug_metrics', action='store_true', help='Print per-trial metrics for debugging')
    parser.add_argument('--binary_force_R2', action='store_true', help='Force R=2 for binary ablation (default keeps R from args)')
    parser.add_argument('--adaptive_max_n', type=int, default=None, help='Max n (R=2n+1) for adaptive mode')
    parser.add_argument('--expA_trials', type=int, default=None, help='Override Exp A trials')
    parser.add_argument('--expB_trials', type=int, default=None, help='Override Exp B trials')
    args = parser.parse_args()

    if args.base_results_dir:
        base = os.path.abspath(args.base_results_dir)
        args.wbit_expA_out = os.path.join(base, 'wbit', 'expA')
        args.wbit_expB_out = os.path.join(base, 'wbit', 'expB')
        args.wbit_expB_grid_out = os.path.join(base, 'wbit', 'expB_grid')
        args.wbit_expC_out = os.path.join(base, 'wbit', 'expC')

        args.binary_expA_out = os.path.join(base, 'binary', 'expA')
        args.binary_expB_out = os.path.join(base, 'binary', 'expB')
        args.binary_expB_grid_out = os.path.join(base, 'binary', 'expB_grid')
        args.binary_expC_out = os.path.join(base, 'binary', 'expC')
        args.adaptive_expA_out = os.path.join(base, 'adaptive', 'expA')
        args.adaptive_expB_out = os.path.join(base, 'adaptive', 'expB')
        args.adaptive_expB_grid_out = os.path.join(base, 'adaptive', 'expB_grid')
        args.adaptive_expC_out = os.path.join(base, 'adaptive', 'expC')

        args.report_out = os.path.join(base, 'phase2_report.csv')

    print("=== w-bit Phase 2 Runner (wbit vs binary scaffold) ===")
    if args.binary_force_R2:
        print("Binary baseline: strict R=2 ablation enabled (--binary_force_R2)")
    else:
        print("Binary baseline: binary-quantized with R matching wbit (use --binary_force_R2 for strict R=2)")

    python_exe = sys.executable
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(base_dir, 'experiments')
    analysis_dir = os.path.join(base_dir, 'analysis')

    def mode_params(mode_label):
        expA_trials = str(args.expA_trials) if args.expA_trials is not None else '100'
        expA_sigma = '0.1'
        expA_grid = '10'
        expA_R = '5'
        expA_density_args = []

        expB_trials = str(args.expB_trials) if args.expB_trials is not None else '100'
        expB_R = '3'

        expC_trials = str(args.expC_trials) if args.expC_trials is not None else '20'
        expC_R = '2'
        expC_population = str(args.expC_population) if args.expC_population is not None else '8'
        expC_elite_k = str(args.expC_elite_k) if args.expC_elite_k is not None else '2'
        expC_restarts = str(args.expC_restarts) if args.expC_restarts is not None else '5'
        expC_max_epochs = str(args.expC_max_epochs) if args.expC_max_epochs is not None else None

        expB_grid_trials = '50'

        if args.smoke:
            expA_trials = '5'
            expA_sigma = '0.1'
            expA_grid = '20'
            expA_density_args = ['--obstacle_density', '0.38']

            expB_trials = '3'

            expC_trials = '3'
            expC_population = '2'
            expC_elite_k = '1'
            expC_restarts = '1'
            expC_max_epochs = '300'

            expB_grid_trials = '3'

        return {
            'expA_trials': expA_trials,
            'expA_sigma': expA_sigma,
            'expA_grid': expA_grid,
            'expA_R': expA_R,
            'expA_density_args': expA_density_args,
            'expB_trials': expB_trials,
            'expB_R': expB_R,
            'expC_trials': expC_trials,
            'expC_R': expC_R,
            'expC_population': expC_population,
            'expC_elite_k': expC_elite_k,
            'expC_restarts': expC_restarts,
            'expC_max_epochs': expC_max_epochs,
            'expB_grid_trials': expB_grid_trials,
            'binary_force_R2': args.binary_force_R2,
            'adaptive_max_n': args.adaptive_max_n,
        }

    def run_suite(mode_label, out_expA, out_expB, out_expB_grid, out_expC):
        params = mode_params(mode_label)

        print(f"\n-- Mode: {mode_label} --")
        print("Running Experiment A: Router Sweep...")
        cmd_a = [
            python_exe, os.path.join(exp_dir, 'exp_a_router_sweep.py'),
            '--trials', params['expA_trials'],
            '--sigma', params['expA_sigma'],
            '--grid', params['expA_grid'],
            '--R', params['expA_R'],
            '--output_dir', out_expA,
            '--mode', mode_label
        ]
        if params.get('binary_force_R2') and mode_label == 'binary':
            cmd_a.append('--binary_force_R2')
        if params['expA_density_args']:
            cmd_a.extend(params['expA_density_args'])
        if params.get('adaptive_max_n') is not None and mode_label == 'adaptive':
            cmd_a.extend(['--adaptive_max_n', str(params['adaptive_max_n'])])
        if args.debug_metrics:
            cmd_a.append('--debug_metrics')
        subprocess.run(cmd_a, check=True, cwd=base_dir)

        print("Running Experiment B: Noise Breakdown...")
        cmd_b = [
            python_exe, os.path.join(exp_dir, 'exp_b_noise_breakdown.py'),
            '--trials', params['expB_trials'],
            '--R', params['expB_R'],
            '--output_dir', out_expB,
            '--mode', mode_label
        ]
        if params.get('binary_force_R2') and mode_label == 'binary':
            cmd_b.append('--binary_force_R2')
        if params.get('adaptive_max_n') is not None and mode_label == 'adaptive':
            cmd_b.extend(['--adaptive_max_n', str(params['adaptive_max_n'])])
        if args.debug_metrics:
            cmd_b.append('--debug_metrics')
        subprocess.run(cmd_b, check=True, cwd=base_dir)

        if args.run_expB_grid:
            print("Running Experiment B Grid Sweep...")
            cmd_bg = [
                python_exe, os.path.join(exp_dir, 'exp_b_weight_noise_grid.py'),
                '--trials', params['expB_grid_trials'],
                '--output_dir', out_expB_grid,
                '--mode', mode_label
            ]
            if params.get('binary_force_R2') and mode_label == 'binary':
                cmd_bg.append('--binary_force_R2')
            if params.get('adaptive_max_n') is not None and mode_label == 'adaptive':
                cmd_bg.extend(['--adaptive_max_n', str(params['adaptive_max_n'])])
            if args.debug_metrics:
                cmd_bg.append('--debug_metrics')
            subprocess.run(cmd_bg, check=True, cwd=base_dir)

        if mode_label != 'adaptive':  # Exp C not yet adapted for adaptive mode
            print("Running Experiment C: Learning Search...")
            cmd_c = [
                python_exe, os.path.join(exp_dir, 'exp_c_learning_search.py'),
                '--trials', params['expC_trials'],
                '--R', params['expC_R'],
                '--output_dir', out_expC,
                '--mode', mode_label
            ]
            if params['expC_population']:
                cmd_c.extend(['--population', params['expC_population']])
            if params['expC_elite_k']:
                cmd_c.extend(['--elite_k', params['expC_elite_k']])
            if params['expC_restarts']:
                cmd_c.extend(['--restarts', params['expC_restarts']])
            if params['expC_max_epochs']:
                cmd_c.extend(['--max_epochs', params['expC_max_epochs']])
            if params.get('adaptive_max_n') is not None and mode_label == 'adaptive':
                cmd_c.extend(['--adaptive_max_n', str(params['adaptive_max_n'])])
            if args.debug_metrics:
                cmd_c.append('--debug_metrics')
            subprocess.run(cmd_c, check=True, cwd=base_dir)

    # Run both modes
    run_suite('wbit', args.wbit_expA_out, args.wbit_expB_out, args.wbit_expB_grid_out, args.wbit_expC_out)
    run_suite('binary', args.binary_expA_out, args.binary_expB_out, args.binary_expB_grid_out, args.binary_expC_out)
    run_suite('adaptive', args.adaptive_expA_out, args.adaptive_expB_out, args.adaptive_expB_grid_out, args.adaptive_expC_out)

    print("\nAggregating Phase 2 report...")
    cmd_report = [
        python_exe, os.path.join(analysis_dir, 'aggregate_phase2_report.py'),
        '--wbit_expA_dir', args.wbit_expA_out,
        '--wbit_expB_dir', args.wbit_expB_grid_out,
        '--wbit_expB_noise_dir', args.wbit_expB_out,
        '--wbit_expC_dir', args.wbit_expC_out,
        '--binary_expA_dir', args.binary_expA_out,
        '--binary_expB_dir', args.binary_expB_grid_out,
        '--binary_expB_noise_dir', args.binary_expB_out,
        '--binary_expC_dir', args.binary_expC_out,
        '--adaptive_expA_dir', args.adaptive_expA_out,
        '--adaptive_expB_dir', args.adaptive_expB_grid_out,
        '--adaptive_expB_noise_dir', args.adaptive_expB_out,
        '--adaptive_expC_dir', args.adaptive_expC_out,
        '--out', args.report_out
    ]
    subprocess.run(cmd_report, check=True, cwd=base_dir)

    print("\n=== Phase 2 scaffold complete ===")
    print(f"Report: {args.report_out}")

if __name__ == "__main__":
    run_phase2()
