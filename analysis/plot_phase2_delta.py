import argparse
import csv
import os
import matplotlib.pyplot as plt

def load_delta(path, weight_scale):
    if not os.path.exists(path):
        print(f"Missing delta file: {path}")
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                ws = float(r.get('weight_scale', 0))
                if ws != weight_scale:
                    continue
                sigma = float(r.get('sigma', 0))
                dsr = float(r.get('delta_success_rate', 0))
                drcp = float(r.get('delta_mean_rcp', 0))
                rows.append((sigma, dsr, drcp))
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda x: x[0])
    return rows

def main(args):
    data = load_delta(args.delta_csv, args.weight_scale)
    if not data:
        print("No data to plot.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    sigmas = [d[0] for d in data]
    delta_sr = [d[1] for d in data]
    delta_rcp = [d[2] for d in data]

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, delta_sr, marker='o')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Delta Success Rate (wbit - binary) vs Sigma (ws={args.weight_scale})')
    plt.xlabel('Sigma')
    plt.ylabel('Delta Success Rate')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'delta_success_rate_vs_sigma.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, delta_rcp, marker='o')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Delta Mean RCP (wbit - binary) vs Sigma (ws={args.weight_scale})')
    plt.xlabel('Sigma')
    plt.ylabel('Delta Mean RCP')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'delta_mean_rcp_vs_sigma.png'))
    plt.close()

    print(f"Delta plots written to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta_csv', type=str, default='results/phase2/phase2_delta_summary.csv')
    parser.add_argument('--output_dir', type=str, default='results/phase2/plots')
    parser.add_argument('--weight_scale', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
