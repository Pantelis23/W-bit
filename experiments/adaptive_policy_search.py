import argparse
import csv
import os
import sys

# Add parent directory to path to import wbit helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import run_single_trial from Experiment B to reuse dynamics
sys.path.append(os.path.dirname(__file__))
try:
    from exp_b_noise_breakdown import run_single_trial
except ImportError as e:
    raise ImportError(
        "Failed to import run_single_trial from exp_b_noise_breakdown. "
        "Run from repo root: `python3 experiments/adaptive_policy_search.py ...`. "
        f"Original error: {e}"
    ) from e


def n_from_policy(sigma, threshold, n_low, n_high, adaptive_max_n):
    n = n_low if sigma < threshold else n_high
    if adaptive_max_n is not None:
        n = min(n, adaptive_max_n)
    return max(1, n)


def evaluate_policy(args, sigma_threshold, n_low, n_high):
    results = {
        "success": 0,
        "trials": 0,
        "rcp": 0.0,
        "R_effective_sum": 0.0,
        "total_samples": 0,
    }

    sigmas = args.sigma_list if args.sigma_list else [round(x * 0.1, 1) for x in range(0, 9)]
    weight_scales = args.weight_scales if args.weight_scales else [1.0, 0.5]

    for ws in weight_scales:
        for sigma in sigmas:
            n_effective = n_from_policy(sigma, sigma_threshold, n_low, n_high, args.adaptive_max_n)
            R_effective = max(3, 2 * n_effective + 1)
            for trial in range(args.trials):
                trial_seed = args.seed + trial + int(sigma * 1000) + int(ws * 10000)
                success, steps, conf, rcp, active_cells = run_single_trial(
                    sigma, ws, trial_seed, R_effective, args.T, args.max_steps,
                    mode='adaptive', n_effective=n_effective, net_mode='adaptive', debug_metrics=args.debug_metrics, debug_trial=getattr(args, 'debug_trial', False)
                )
                results["trials"] += 1
                if success:
                    results["success"] += 1
                results["rcp"] += rcp
                results["R_effective_sum"] += R_effective
                results["total_samples"] += 1

    success_rate = results["success"] / results["trials"] if results["trials"] > 0 else 0.0
    avg_rcp = results["rcp"] / results["trials"] if results["trials"] > 0 else 0.0
    avg_R_effective = results["R_effective_sum"] / results["trials"] if results["trials"] > 0 else 0.0

    # Normalize RCP by R^2 to keep scoring stable across radix
    rcp_norm = avg_rcp / (args.R * args.R) if args.R > 0 else avg_rcp

    score = success_rate - (args.alpha * avg_R_effective) - (args.beta * rcp_norm)

    return {
        "sigma_threshold": sigma_threshold,
        "n_low": n_low,
        "n_high": n_high,
        "success_rate": success_rate,
        "avg_rcp": avg_rcp,
        "avg_R_effective": avg_R_effective,
        "score": score,
        "trials": results["trials"],
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search simple adaptive policies over sigma thresholds.")
    parser.add_argument('--sigma_thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7], help='Threshold candidates for switching n_effective')
    parser.add_argument('--n_high_candidates', type=int, nargs='+', default=[2, 3], help='High n values to test when sigma >= threshold')
    parser.add_argument('--n_low', type=int, default=1, help='Low n value when sigma < threshold')
    parser.add_argument('--adaptive_max_n', type=int, default=None, help='Clamp n for adaptive mode')
    parser.add_argument('--sigma_list', type=float, nargs='+', default=None, help='Sigma sweep used to score policies')
    parser.add_argument('--weight_scales', type=float, nargs='+', default=None, help='Weight scale sweep used to score policies')
    parser.add_argument('--trials', type=int, default=10, help='Trials per sigma/weight_scale')
    parser.add_argument('--seed', type=int, default=99, help='Base seed')
    parser.add_argument('--R', type=int, default=3, help='Base R for scoring / normalization')
    parser.add_argument('--T', type=float, default=0.2)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.05, help='Penalty weight on avg_R_effective')
    parser.add_argument('--beta', type=float, default=0.001, help='Penalty weight on normalized RCP')
    parser.add_argument('--output_csv', type=str, default='results/phase2/adaptive_policy_search.csv')
    parser.add_argument('--debug_metrics', action='store_true')
    parser.add_argument('--debug_trial', action='store_true')
    parser.add_argument('--print_metrics', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.exists(args.output_csv)

    best = None
    rows = []

    for sigma_t in args.sigma_thresholds:
        for n_high in args.n_high_candidates:
            stats = evaluate_policy(args, sigma_t, args.n_low, n_high)
            rows.append(stats)
            if best is None or stats['score'] > best['score']:
                best = stats
            if args.print_metrics:
                print({
                    "sigma_threshold": sigma_t,
                    "n_low": args.n_low,
                    "n_high": n_high,
                    "success_rate": stats['success_rate'],
                    "avg_R_effective": stats['avg_R_effective'],
                    "avg_rcp": stats['avg_rcp'],
                    "score": stats['score'],
                })

    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        header = ['policy_id', 'sigma_threshold', 'n_low', 'n_high', 'success_rate', 'avg_R_effective', 'avg_rcp', 'score', 'trials']
        if write_header:
            writer.writerow(header)
        for idx, row in enumerate(rows):
            csv_row = [
                idx, row['sigma_threshold'], row['n_low'], row['n_high'],
                f"{row['success_rate']:.4f}", f"{row['avg_R_effective']:.2f}", f"{row['avg_rcp']:.2f}", f"{row['score']:.4f}", row['trials']
            ]
            writer.writerow(csv_row)

    if best is not None:
        print("\n=== Best Policy (adaptive_policy_v1 candidate) ===")
        print(f"threshold: {best['sigma_threshold']}, n_low: {best['n_low']}, n_high: {best['n_high']}")
        print(f"success_rate: {best['success_rate']:.4f}, avg_R_effective: {best['avg_R_effective']:.2f}, avg_rcp: {best['avg_rcp']:.2f}, score: {best['score']:.4f}")


if __name__ == "__main__":
    main()
