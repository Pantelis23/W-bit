import argparse
import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def load_results_csv(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def group_by_sigma(rows, weight_scale):
    grouped = defaultdict(list)
    for r in rows:
        try:
            ws = float(r.get('weight_scale', 0))
            sigma = float(r.get('sigma', 0))
            success = float(r.get('success', 0))
            rcp = float(r.get('rcp', 0))
        except (TypeError, ValueError):
            continue
        if ws == weight_scale:
            grouped[sigma].append((success, rcp))
    sigmas = sorted(grouped.keys())
    result = []
    for s in sigmas:
        vals = grouped[s]
        n = len(vals)
        if n == 0:
            continue
        success_rate = sum(v[0] for v in vals) / n
        avg_rcp = sum(v[1] for v in vals) / n
        result.append((s, success_rate, avg_rcp))
    return result

def plot_phase2(exp_dir_wbit, exp_dir_binary, output_dir, weight_scale=1.0):
    wbit_path = os.path.join(exp_dir_wbit, 'results.csv')
    binary_path = os.path.join(exp_dir_binary, 'results.csv')

    rows_w = load_results_csv(wbit_path)
    rows_b = load_results_csv(binary_path)
    if not rows_w or not rows_b:
        print("Missing input data for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    data_w = group_by_sigma(rows_w, weight_scale)
    data_b = group_by_sigma(rows_b, weight_scale)

    sig_w, sr_w, rcp_w = zip(*data_w) if data_w else ([], [], [])
    sig_b, sr_b, rcp_b = zip(*data_b) if data_b else ([], [], [])

    plt.figure(figsize=(8, 5))
    plt.plot(sig_w, sr_w, marker='o', label='wbit')
    plt.plot(sig_b, sr_b, marker='o', label='binary')
    plt.title(f'Phase 2: Success Rate vs Sigma (weight_scale={weight_scale})')
    plt.xlabel('Sigma')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'success_rate_vs_sigma.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(sig_w, rcp_w, marker='o', label='wbit')
    plt.plot(sig_b, rcp_b, marker='o', label='binary')
    plt.title(f'Phase 2: Avg RCP vs Sigma (weight_scale={weight_scale})')
    plt.xlabel('Sigma')
    plt.ylabel('Avg RCP')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'avg_rcp_vs_sigma.png'))
    plt.close()

    print(f"Plots written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wbit_dir', type=str, default='results/phase2/wbit/expB')
    parser.add_argument('--binary_dir', type=str, default='results/phase2/binary/expB')
    parser.add_argument('--output_dir', type=str, default='results/phase2/plots')
    parser.add_argument('--weight_scale', type=float, default=1.0)
    args = parser.parse_args()

    plot_phase2(args.wbit_dir, args.binary_dir, args.output_dir, weight_scale=args.weight_scale)
