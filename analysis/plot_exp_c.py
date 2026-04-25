import os
import csv
import argparse
import matplotlib.pyplot as plt

def plot_exp_c(results_dir, output_dir):
    print(f"Loading Exp C results from {results_dir}...")
    csv_path = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                H = int(r.get('H', 0))
                allow_direct = r.get('allow_direct_ab_to_y')
                success = float(r.get('success', 0))
            except (TypeError, ValueError):
                continue
            rows.append((H, allow_direct, success))

    if not rows:
        print("No data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by H and allow_direct_ab_to_y
    grouped = {}
    for H, allow_direct, success in rows:
        key = (H, allow_direct)
        grouped.setdefault(key, []).append(success)

    # Build data for stacked bars
    H_values = sorted({k[0] for k in grouped.keys()})
    direct_flags = sorted({k[1] for k in grouped.keys()})
    data = {flag: [] for flag in direct_flags}
    for H in H_values:
        for flag in direct_flags:
            vals = grouped.get((H, flag), [])
            avg = sum(vals) / len(vals) if vals else 0.0
            data[flag].append(avg)

    x = range(len(H_values))
    plt.figure(figsize=(10, 6))
    width = 0.35 if len(direct_flags) == 2 else 0.6
    for idx, flag in enumerate(direct_flags):
        offsets = [xi + (idx - len(direct_flags)/2)*width for xi in x]
        plt.bar(offsets, data[flag], width=width, label=f'allow_direct_ab_to_y={flag}')

    plt.title('Learning Success Rate vs Hidden Neurons')
    plt.xlabel('Hidden Neurons (H)')
    plt.ylabel('Success Rate')
    plt.grid(True, axis='y')
    plt.ylim(0, 1.1)
    plt.xticks(x, [str(h) for h in H_values])
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learning_success_vs_H.png'))
    plt.close()
    
    print(f"Plot saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/expC')
    parser.add_argument('--output_dir', type=str, default='results/expC/plots')
    args = parser.parse_args()
    
    try:
        plot_exp_c(args.results_dir, args.output_dir)
    except ImportError:
        print("pandas or matplotlib not installed.")
