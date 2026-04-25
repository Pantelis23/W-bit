import argparse
import csv
import os
import matplotlib.pyplot as plt


def load_frontier(path):
    if not os.path.exists(path):
        print(f"Missing frontier CSV: {path}")
        return [], None
    rows = []
    target_success = None
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                sigma = float(r.get('sigma', 0))
                rcp_w = float(r.get('avg_rcp_wbit', 0))
                rcp_b = float(r.get('avg_rcp_binary', 0))
                delta = float(r.get('delta_avg_rcp', 0))
                mt_w = int(r.get('meets_target_wbit', 0)) if r.get('meets_target_wbit') not in (None, '') else 0
                mt_b = int(r.get('meets_target_binary', 0)) if r.get('meets_target_binary') not in (None, '') else 0
                rcp_a = None
                mt_a = 0
                delta_a = None
                if r.get('avg_rcp_adaptive') not in (None, ''):
                    rcp_a = float(r.get('avg_rcp_adaptive', 0))
                    mt_a = int(r.get('meets_target_adaptive', 0)) if r.get('meets_target_adaptive') not in (None, '') else 0
                if r.get('delta_avg_rcp_adaptive') not in (None, ''):
                    delta_a = float(r.get('delta_avg_rcp_adaptive', 0))
                if target_success is None and r.get('target_success') not in (None, ''):
                    target_success = float(r.get('target_success'))
                rows.append((sigma, rcp_w, rcp_b, delta, mt_w, mt_b, rcp_a, delta_a, mt_a))
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda x: x[0])
    return rows, target_success


def main(args):
    data, target_success = load_frontier(args.frontier_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    if not data:
        print("No frontier data to plot.")
        # Still emit placeholder plots for visibility
        plt.figure(figsize=(8, 5))
        plt.title('Phase 2 Frontier: Mean RCP vs Sigma (no data)')
        plt.savefig(os.path.join(args.output_dir, 'frontier_mean_rcp_vs_sigma.png'))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.title('Phase 2 Frontier: Δ Mean RCP vs Sigma (no data)')
        plt.savefig(os.path.join(args.output_dir, 'frontier_delta_mean_rcp_vs_sigma.png'))
        plt.close()
        return

    sigmas = [d[0] for d in data]
    rcp_w = [d[1] for d in data]
    rcp_b = [d[2] for d in data]
    delta = [d[3] for d in data]
    rcp_a = [d[6] for d in data if d[6] is not None]
    has_adaptive = any(d[6] is not None for d in data)
    delta_a = [d[7] for d in data if d[7] is not None] if has_adaptive else []
    ts_suffix = f" (target_success={target_success:.2f})" if target_success is not None else ""

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, rcp_w, marker='o', label='wbit frontier')
    plt.plot(sigmas, rcp_b, marker='o', label='binary frontier')
    if has_adaptive:
        plt.plot(sigmas, [d[6] if d[6] is not None else float('nan') for d in data], marker='o', label='adaptive frontier')
    plt.title(f'Phase 2 Frontier: Mean RCP vs Sigma{ts_suffix}')
    plt.xlabel('Sigma')
    plt.ylabel('Mean RCP at target success')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'frontier_mean_rcp_vs_sigma.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, delta, marker='o', label='wbit - binary')
    if has_adaptive:
        plt.plot(sigmas, [d[7] if d[7] is not None else float('nan') for d in data], marker='o', label='wbit - adaptive')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Phase 2 Frontier: Δ Mean RCP vs Sigma{ts_suffix}')
    plt.xlabel('Sigma')
    plt.ylabel('Delta Mean RCP')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'frontier_delta_mean_rcp_vs_sigma.png'))
    plt.close()

    print(f"Frontier plots written to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frontier_csv', type=str, default='results/phase2/phase2_frontier_summary.csv')
    parser.add_argument('--output_dir', type=str, default='results/phase2/plots')
    args = parser.parse_args()
    main(args)
