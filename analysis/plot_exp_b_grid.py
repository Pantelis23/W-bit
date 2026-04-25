import os
import csv
import argparse
import matplotlib.pyplot as plt

def plot_exp_b_grid(results_dir, output_dir):
    print(f"Loading summary.csv from {results_dir}...")
    summary_path = os.path.join(results_dir, 'summary.csv')
    
    if not os.path.exists(summary_path):
        print(f"File not found: {summary_path}")
        return

    rows = []
    with open(summary_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                ws = float(r.get('weight_scale', 0))
                sigma = float(r.get('sigma', 0))
                success_rate = float(r.get('success_rate', 0))
                avg_rcp = float(r.get('avg_rcp', 0))
            except (TypeError, ValueError):
                continue
            rows.append((ws, sigma, success_rate, avg_rcp))

    if not rows:
        print("No valid data loaded.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    weight_scales = sorted({r[0] for r in rows}, reverse=True)
    
    # 1. Success Rate vs Sigma
    plt.figure(figsize=(10, 6))
    for ws in weight_scales:
        sigmas = sorted([r[1] for r in rows if r[0] == ws])
        success_vals = []
        for s in sigmas:
            vals = [r[2] for r in rows if r[0] == ws and r[1] == s]
            if vals:
                success_vals.append((s, sum(vals)/len(vals)))
        if success_vals:
            xs = [v[0] for v in success_vals]
            ys = [v[1] for v in success_vals]
            plt.plot(xs, ys, marker='o', label=f'Weight Scale {ws}')
        
    plt.title('Noise Breakdown: Success Rate vs Sigma')
    plt.xlabel('Sigma (Noise Level)')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'success_rate_vs_sigma.png'))
    plt.close()
    
    # 2. Avg RCP vs Sigma
    plt.figure(figsize=(10, 6))
    for ws in weight_scales:
        sigmas = sorted([r[1] for r in rows if r[0] == ws])
        rcp_vals = []
        for s in sigmas:
            vals = [r[3] for r in rows if r[0] == ws and r[1] == s]
            if vals:
                rcp_vals.append((s, sum(vals)/len(vals)))
        if rcp_vals:
            xs = [v[0] for v in rcp_vals]
            ys = [v[1] for v in rcp_vals]
            plt.plot(xs, ys, marker='o', label=f'Weight Scale {ws}')
        
    plt.title('Noise Breakdown: Avg RCP vs Sigma')
    plt.xlabel('Sigma (Noise Level)')
    plt.ylabel('Avg RCP')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'avg_rcp_vs_sigma.png'))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/expB_grid')
    parser.add_argument('--output_dir', type=str, default='results/expB_grid/plots')
    args = parser.parse_args()
    
    try:
        plot_exp_b_grid(args.results_dir, args.output_dir)
    except ImportError:
        print("pandas or matplotlib not installed. Skipping plotting.")
