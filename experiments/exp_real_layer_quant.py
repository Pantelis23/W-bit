import argparse
import csv
import os
import random
import sys

# Add parent directory to path to import wbit
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit.quantization_lab import wbit_eval_layer


def load_from_npz(path):
    try:
        import numpy as np  # Lazy import to avoid hard dependency when unused
    except ImportError as exc:
        raise ImportError("numpy is required to load --weights_npz inputs") from exc
    data = np.load(path)
    if 'W' not in data or 'x_batch' not in data:
        raise ValueError("npz file must contain arrays named 'W' and 'x_batch'")
    W = data['W']
    x_batch = data['x_batch']
    return W, x_batch


def generate_demo(seed, in_dim, out_dim, batch_size):
    random.seed(seed)
    W = [[random.gauss(0, 0.5) for _ in range(in_dim)] for _ in range(out_dim)]
    x_batch = []
    for _ in range(batch_size):
        x = [random.gauss(0, 1.0) for _ in range(in_dim)]
        x_batch.append(x)
    return W, x_batch


def main():
    parser = argparse.ArgumentParser(description="Quantize a real layer into W-bit states and measure degradation.")
    parser.add_argument('--weights_npz', type=str, help="Optional .npz with arrays 'W' and 'x_batch' for real-model eval")
    parser.add_argument('--layer_name', type=str, default='layer', help='Name recorded in CSV output')
    parser.add_argument('--mode', type=str, default='wbit', choices=['wbit', 'binary', 'adaptive'])
    parser.add_argument('--R', type=int, default=3, help='Configured radix for quantization')
    parser.add_argument('--sigma', type=float, default=0.0, help='Noise stddev applied to quantized weights')
    parser.add_argument('--sigma_list', type=float, nargs='+', default=None, help='Optional sweep over multiple sigmas')
    parser.add_argument('--trials', type=int, default=10, help='Noisy trials to average')
    parser.add_argument('--binary_force_R2', action='store_true', help='Strict binary ablation (R_effective = 2)')
    parser.add_argument('--adaptive_max_n', type=int, default=None, help='Clamp for adaptive mode (R=2n+1)')
    parser.add_argument('--output_csv', type=str, default='results/phase2/real_layer_quant.csv')
    parser.add_argument('--seed', type=int, default=123, help='Seed for demo generation when no npz is provided')
    parser.add_argument('--demo_in_dim', type=int, default=8, help='Demo input dimension when generating synthetic weights')
    parser.add_argument('--demo_out_dim', type=int, default=3, help='Demo output dimension when generating synthetic weights')
    parser.add_argument('--demo_batch', type=int, default=32, help='Demo batch size when generating synthetic weights')
    parser.add_argument('--print_metrics', action='store_true', help='Print per-sigma metrics for quick inspection')
    args = parser.parse_args()

    if args.weights_npz:
        W, x_batch = load_from_npz(args.weights_npz)
    else:
        W, x_batch = generate_demo(args.seed, args.demo_in_dim, args.demo_out_dim, args.demo_batch)

    sigmas = args.sigma_list if args.sigma_list else [args.sigma]
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.exists(args.output_csv)

    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        header = [
            'layer_name', 'mode', 'mode_variant', 'net_mode', 'R', 'R_effective', 'n_effective',
            'sigma', 'trials', 'success_rate', 'loss_delta', 'avg_rcp'
        ]
        if write_header:
            writer.writerow(header)

        for sigma in sigmas:
            stats = wbit_eval_layer(
                W, x_batch, mode=args.mode, R=args.R, sigma=sigma,
                trials=args.trials, binary_force_R2=args.binary_force_R2, adaptive_max_n=args.adaptive_max_n
            )
            row = [
                args.layer_name, stats['mode'], stats['mode_variant'], stats['net_mode'],
                args.R, stats['R_effective'], stats['n_effective'], stats['sigma'], stats['trials'],
                f"{stats['success_rate']:.4f}", f"{stats['loss_delta']:.6f}", f"{stats['avg_rcp']:.1f}"
            ]
            writer.writerow(row)
            if args.print_metrics:
                print({
                    "layer": args.layer_name,
                    "mode": stats['mode'],
                    "mode_variant": stats['mode_variant'],
                    "sigma": stats['sigma'],
                    "R_effective": stats['R_effective'],
                    "n_effective": stats['n_effective'],
                    "success_rate": stats['success_rate'],
                    "loss_delta": stats['loss_delta'],
                    "avg_rcp": stats['avg_rcp'],
                })


if __name__ == "__main__":
    main()
