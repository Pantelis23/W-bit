import argparse
import sys
import os
import random
import csv
import statistics

# Add parent directory to path to import wbit
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import run_single_trial from sibling script by adding current dir to path
sys.path.append(os.path.dirname(__file__))
try:
    from exp_b_noise_breakdown import run_single_trial
except ImportError as e:
    raise ImportError(
        "Failed to import run_single_trial from exp_b_noise_breakdown. "
        "Run from repo root: `python3 experiments/exp_b_weight_noise_grid.py ...`. "
        "This grid runner depends on the shared helper. "
        f"Original error: {e}"
    ) from e

def run_grid_sweep(args):
    print("=== Experiment B Weight/Noise Grid Runner (Phase 1.8) ===")
    
    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv_path = os.path.join(args.output_dir, 'summary.csv')
    phase_csv_path = os.path.join(args.output_dir, 'phase_diagram.csv')
    
    if args.weight_scales:
        weight_scales = args.weight_scales
    else:
        weight_scales = [1.0, 0.5, 0.2, 0.1, 0.05]
        
    if args.sigma_list:
        sigmas = args.sigma_list
    else:
        sigmas = [round(x * 0.1, 1) for x in range(0, 21)]

    r_values = args.R_values if args.R_values else [args.R]
        
    # Store aggregated data for post-hoc breakpoint analysis
    agg_data = {} # R -> ws -> {sigma -> success_rate}

    with open(summary_csv_path, 'w', newline='') as f_sum, open(phase_csv_path, 'w', newline='') as f_phase:
        writer = csv.writer(f_sum)
        header = ['weight_scale', 'sigma', 'trials', 'success_rate', 'avg_steps', 'avg_rcp', 'avg_final_conf', 'R', 'T', 'max_steps', 'mode_variant', 'n_effective', 'R_effective']
        writer.writerow(header)

        phase_writer = csv.writer(f_phase)
        phase_header = ['weight_scale', 'sigma', 'R_effective', 'mode', 'mode_variant', 'n_effective', 'phase_label', 'success_rate', 'avg_rcp']
        phase_writer.writerow(phase_header)

        base_max_n = getattr(args, 'adaptive_max_n', None)
        sigma_threshold = getattr(args, 'adaptive_sigma_threshold', None)
        n_low = getattr(args, 'adaptive_n_low', None)
        n_high = getattr(args, 'adaptive_n_high', None)

        def choose_effective(sigma, base_R):
            if args.mode == 'binary':
                r_eff = 2 if getattr(args, 'binary_force_R2', False) else base_R
                n_eff = max(1, (r_eff - 1) // 2)
                variant = 'binary_strict_R2' if getattr(args, 'binary_force_R2', False) else 'binary_quantized'
                net_mode = 'binary' if r_eff == 2 else 'wbit'
                return r_eff, n_eff, variant, net_mode
            if args.mode == 'adaptive':
                if sigma_threshold is not None and n_low is not None and n_high is not None:
                    n = n_low if sigma is None or sigma < sigma_threshold else n_high
                else:
                    score = 0
                    if sigma is not None and sigma >= 0.5:
                        score += 1
                    n = 1 if score <= 0 else 2 if score == 1 else 3 if score == 2 else 4
                max_n = base_max_n if base_max_n is not None else max(1, (base_R - 1) // 2)
                n = max(1, min(n, max_n))
                r_eff = max(3, 2 * n + 1)
                return r_eff, n, f"adaptive_heuristic_n{n}", 'adaptive'
            r_eff = base_R
            n_eff = max(1, (r_eff - 1) // 2)
            return r_eff, n_eff, 'wbit', 'wbit'

        for base_R in r_values:
            agg_data[base_R] = {}
            print(f"\n=== Sweeping R={base_R} ===")

            for ws in weight_scales:
                print(f"\n--- Running Weight Scale {ws} ---")
                agg_data[base_R][ws] = {}
                
                for sigma in sigmas:
                    success_count = 0
                    total_steps = 0
                    total_rcp = 0.0
                    total_conf = 0.0
                    effective_R, n_effective, mode_variant, net_mode = choose_effective(sigma, base_R)
                    if getattr(args, 'debug_effective', False):
                        print({"mode": args.mode, "weight_scale": ws, "sigma": sigma, "mode_variant": mode_variant, "n_effective": n_effective, "R_effective": effective_R})
                    
                    for trial in range(args.trials):
                        trial_seed = args.seed + trial + int(sigma * 1000) + int(ws * 10000) + int(base_R * 100000)
                        success, steps, conf, rcp, active_cells = run_single_trial(
                            sigma, ws, trial_seed, effective_R, args.T, args.max_steps,
                            mode=args.mode, n_effective=n_effective, net_mode=net_mode, debug_metrics=args.debug_metrics, debug_trial=getattr(args, 'debug_trial', False)
                        )
                        if success: success_count += 1
                        total_steps += steps
                        total_rcp += rcp
                        total_conf += conf
                        
                    success_rate = success_count / args.trials
                    avg_steps = total_steps / args.trials
                    avg_rcp = total_rcp / args.trials
                    avg_final_conf = total_conf / args.trials
                    
                    agg_data[base_R][ws][sigma] = success_rate
                    
                    row = [ws, sigma, args.trials, f"{success_rate:.2f}", f"{avg_steps:.1f}", f"{avg_rcp:.1f}", f"{avg_final_conf:.3f}", effective_R, args.T, args.max_steps, mode_variant, n_effective, effective_R]
                    assert len(row) == len(header), f"Row/Header mismatch: {len(row)} vs {len(header)}"
                    writer.writerow(row)

                    if success_rate >= 0.90:
                        phase_label = 'good'
                    elif success_rate >= 0.10:
                        phase_label = 'edge'
                    else:
                        phase_label = 'fail'
                    phase_row = [ws, sigma, effective_R, args.mode, mode_variant, n_effective, phase_label, f"{success_rate:.2f}", f"{avg_rcp:.1f}"]
                    assert len(phase_row) == len(phase_header), f"Phase Row/Header mismatch: {len(phase_row)} vs {len(phase_header)}"
                    phase_writer.writerow(phase_row)
        
        # B4: Robust Breakpoint Extraction
        print("\n=== Breakpoint Summary ===")
        print(f"{ 'R':<5} { 'Weight Scale':<15} { 'Sigma (90%)':<15} { 'Sigma (50%)':<15}")
        print("-" * 60)
        for base_R in r_values:
            for ws in weight_scales:
                s90 = -1.0
                s50 = -1.0
                
                # Iterate sorted sigmas to find max thresholds
                for sigma in sigmas:
                    rate = agg_data[base_R][ws].get(sigma, 0.0)
                    if rate >= 0.90: s90 = sigma
                    if rate >= 0.50: s50 = sigma
                
                s90_str = f"{s90:.1f}" if s90 >= 0 else "None"
                s50_str = f"{s50:.1f}" if s50 >= 0 else "None"
                
                print(f"{base_R:<5} {ws:<15} {s90_str:<15} {s50_str:<15}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/expB_grid')
    
    parser.add_argument('--R', type=int, default=3)
    parser.add_argument('--R_values', type=int, nargs='+', help='Optional sweep over multiple R values (overrides --R)', default=None)
    parser.add_argument('--T', type=float, default=0.2)
    parser.add_argument('--max_steps', type=int, default=50)
    
    parser.add_argument('--weight_scales', type=float, nargs='+', help='List of weight scales', default=None)
    parser.add_argument('--sigma_list', type=float, nargs='+', help='List of sigmas', default=None)
    parser.add_argument('--mode', type=str, default='wbit', choices=['wbit', 'binary', 'adaptive'], help='Logic substrate mode (wbit default, binary placeholder)')
    parser.add_argument('--debug_metrics', action='store_true')
    parser.add_argument('--binary_force_R2', action='store_true', help='Force R=2 in binary mode (strict ablation)')
    parser.add_argument('--adaptive_max_n', type=int, default=None, help='Max n (R=2n+1) for adaptive mode')
    parser.add_argument('--adaptive_sigma_threshold', type=float, default=None, help='Optional sigma threshold policy override for adaptive mode')
    parser.add_argument('--adaptive_n_low', type=int, default=None, help='n when sigma < threshold (adaptive override)')
    parser.add_argument('--adaptive_n_high', type=int, default=None, help='n when sigma >= threshold (adaptive override)')
    parser.add_argument('--debug_effective', action='store_true', help='Print effective R/n per sigma for debugging')
    parser.add_argument('--debug_trial', action='store_true', help='Print per-trial effective params/results')
    
    args = parser.parse_args()
    run_grid_sweep(args)
