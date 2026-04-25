import argparse
import sys
import os
import random
import csv
import statistics

# Add parent directory to path to import wbit
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit.analog_network import AnalogWbitNetwork
from wbit.step_utils import summarize_steps


def compute_adaptive_n(grid, density, sigma, optimal_dist, adaptive_max_n, fallback_R, sigma_threshold=None, n_low=None, n_high=None):
    """
    Default heuristic unless an explicit sigma threshold policy is provided.
    sigma_threshold + n_low/n_high implements an adaptive_policy_v1-style controller.
    """
    if sigma_threshold is not None and n_low is not None and n_high is not None:
        n = n_low if sigma is None or sigma < sigma_threshold else n_high
    else:
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
            n = 1
        elif score == 1:
            n = 2
        elif score == 2:
            n = 3
        else:
            n = 4
    max_n = adaptive_max_n if adaptive_max_n is not None else max(1, (fallback_R - 1) // 2)
    n = max(1, min(n, max_n))
    return n

def run_single_trial(sigma, weight_scale, trial_seed, R, T, max_steps, mode='wbit', n_effective=None, net_mode=None, debug_metrics=False, debug_trial=False):
    random.seed(trial_seed)
    
    # B2: Explicit active cells
    num_cells = 4
    active_cells = num_cells
    
    effective_R = R
    chosen_mode = net_mode if net_mode else mode
    if mode == 'binary' and R != 2 and chosen_mode == 'binary':
        # Quantized baseline uses wbit dynamics with binary labeling when R>2
        chosen_mode = 'wbit'
    net = AnalogWbitNetwork(num_cells=num_cells, R=effective_R, mode=chosen_mode)
    
    # Weights
    ALPHA = 5.0 * weight_scale
    BETA = 20.0 * weight_scale
    
    C, D, O1, O2 = 0, 1, 2, 3
    
    def zeros(rows, cols):
        return [[0.0] * cols for _ in range(rows)]

    def eye(n, scale=1.0):
        mat = zeros(n, n)
        for i in range(n):
            mat[i][i] = scale
        return mat
    
    # D -> O1, O2 (Copy)
    net.set_interaction_weights(O1, D, eye(effective_R, ALPHA))
    net.set_interaction_weights(O2, D, eye(effective_R, ALPHA))
    
    # C -> O1 (Gate) - If C=1, Veto O1 (Force 0)
    w_gate_o1 = zeros(effective_R, effective_R)
    w_gate_o1[0][1] = BETA
    net.set_interaction_weights(O1, C, w_gate_o1)
    
    # C -> O2 (Gate) - If C=0, Veto O2 (Force 0)
    w_gate_o2 = zeros(effective_R, effective_R)
    w_gate_o2[0][0] = BETA
    net.set_interaction_weights(O2, C, w_gate_o2)
    
    # Random Case
    c_val = random.randint(0, 1)
    d_val = random.randint(1, effective_R-1)
    
    # Input Biases (Strong Voltage)
    net.reset_local_weights()
    
    c_bias = [-5.0] * effective_R
    c_bias[c_val] = 10.0
    net.set_local_weights(C, c_bias)
    
    d_bias = [-5.0] * effective_R
    d_bias[d_val] = 10.0
    net.set_local_weights(D, d_bias)
    
    # Run
    steps_converged = net.run_until_stable(max_steps=max_steps, temperature=T, noise=sigma)
    
    # Verify
    hard_states = net.get_hard_state()
    o1_res = hard_states[O1]
    o2_res = hard_states[O2]
    
    exp_o1 = d_val if c_val == 0 else 0
    exp_o2 = d_val if c_val == 1 else 0
    
    logic_success = (o1_res == exp_o1) and (o2_res == exp_o2)
    
    # Confidence
    prob_o1 = net.state[O1][o1_res]
    prob_o2 = net.state[O2][o2_res]
    final_conf = min(prob_o1, prob_o2)
    
    success = logic_success and (final_conf >= 0.9)
    
    steps_taken_raw, reason = summarize_steps(success, steps_converged, max_steps, path_exists=True, converged=(steps_converged < max_steps))
    steps_taken = steps_taken_raw
    if success:
        steps_taken = max(steps_taken_raw, 1)
    
    rcp = net.calculate_RCP(steps_taken, i_out=1.0, n_active_cells=active_cells)
    
    if debug_metrics or debug_trial:
        print({
            "mode": mode,
            "sigma": sigma,
            "weight_scale": weight_scale,
            "R_effective": effective_R,
            "n_effective": n_effective,
            "success": success,
            "steps": steps_taken,
            "final_confidence": final_conf,
            "rcp": rcp,
            "reason": reason,
            "o1_res": o1_res,
            "o2_res": o2_res,
            "exp_o1": exp_o1,
            "exp_o2": exp_o2,
            "c_val": c_val,
            "d_val": d_val
        })
    return success, steps_taken, final_conf, rcp, active_cells

def run_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'results.csv')
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        header = ['trial', 'trial_seed', 'sigma', 'snr', 'success', 'steps', 'final_confidence', 'rcp', 'active_cells', 'weight_scale', 'R', 'T', 'max_steps', 'mode', 'mode_variant', 'n_effective', 'R_effective']
        if write_header:
            writer.writerow(header)
            
        random.seed(args.seed)
        
        sigma_min = getattr(args, 'sigma_min', 0.0)
        if args.sigma is not None:
            sigmas = [args.sigma]
        else:
            steps = int((args.sigma_max - sigma_min) / args.sigma_step)
            sigmas = [round(sigma_min + i * args.sigma_step, 3) for i in range(0, steps + 1)]
        
        print(f"Starting Experiment B: Noise Breakdown")
        print(f"Sigmas: {sigmas}")
        print(f"Weight Scale: {args.weight_scale}")
        print(f"Trials per sigma: {args.trials}")
        print(f"Base Seed: {args.seed}")
        
        adaptive_max_n = getattr(args, 'adaptive_max_n', None)
        sigma_threshold = getattr(args, 'adaptive_sigma_threshold', None)
        n_low = getattr(args, 'adaptive_n_low', None)
        n_high = getattr(args, 'adaptive_n_high', None)
        for sigma in sigmas:
            if args.mode == 'binary':
                effective_R = 2 if getattr(args, 'binary_force_R2', False) else args.R
                mode_variant = 'binary_strict_R2' if getattr(args, 'binary_force_R2', False) else 'binary_quantized'
                n_effective = max(1, (effective_R - 1) // 2)
                net_mode = 'binary' if effective_R == 2 else 'wbit'
            elif args.mode == 'adaptive':
                n_effective = compute_adaptive_n(None, None, sigma, None, adaptive_max_n, args.R, sigma_threshold, n_low, n_high)
                effective_R = max(3, 2 * n_effective + 1)
                mode_variant = f"adaptive_heuristic_n{n_effective}"
                net_mode = 'adaptive'
            else:
                effective_R = args.R
                n_effective = max(1, (effective_R - 1) // 2)
                mode_variant = 'wbit'
                net_mode = 'wbit'
            success_count = 0
            
            for trial in range(args.trials):
                trial_seed = args.seed + trial + int(sigma * 1000)
                
                success, steps, conf, rcp, active_cells = run_single_trial(
                    sigma, args.weight_scale, trial_seed, effective_R, args.T, args.max_steps,
                    mode=args.mode, n_effective=n_effective, net_mode=net_mode, debug_metrics=args.debug_metrics, debug_trial=getattr(args, 'debug_trial', False)
                )
                
                snr = 1.0 / sigma if sigma > 0 else 999.0
                
                row = [trial, trial_seed, sigma, f"{snr:.2f}", 1 if success else 0, steps, f"{conf:.4f}", f"{rcp:.2f}", active_cells, args.weight_scale, effective_R, args.T, args.max_steps, args.mode, mode_variant, n_effective, effective_R]
                assert len(row) == len(header), f"Row/Header mismatch: {len(row)} vs {len(header)}"
                writer.writerow(row)
                
                if success: success_count += 1
                
            print(f"Sigma {sigma}: Success Rate {success_count/args.trials:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--R', type=int, default=3)
    parser.add_argument('--T', type=float, default=0.2)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results/expB')
    parser.add_argument('--weight_scale', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='wbit', choices=['wbit', 'binary', 'adaptive'], help='Logic substrate mode (wbit default, binary placeholder)')
    parser.add_argument('--sigma_min', type=float, default=0.0)
    parser.add_argument('--sigma_max', type=float, default=2.0)
    parser.add_argument('--sigma_step', type=float, default=0.1)
    parser.add_argument('--debug_metrics', action='store_true')
    parser.add_argument('--binary_force_R2', action='store_true', help='Force R=2 in binary mode (strict ablation)')
    parser.add_argument('--adaptive_max_n', type=int, default=None, help='Max n (R=2n+1) for adaptive mode')
    parser.add_argument('--adaptive_sigma_threshold', type=float, default=None, help='Optional sigma threshold policy override for adaptive mode')
    parser.add_argument('--adaptive_n_low', type=int, default=None, help='n when sigma < threshold (adaptive override)')
    parser.add_argument('--adaptive_n_high', type=int, default=None, help='n when sigma >= threshold (adaptive override)')
    parser.add_argument('--debug_trial', action='store_true', help='Print per-trial effective params/results')
    
    args = parser.parse_args()
    run_experiment(args)
