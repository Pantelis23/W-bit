import argparse
import csv
import os
import matplotlib.pyplot as plt


def load_pareto(path):
    rows = []
    if not os.path.exists(path):
        print(f"Missing pareto file: {path}")
        return rows
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                sigma = float(r.get('sigma', 0))
                ws = float(r.get('weight_scale', 0))
                sr = float(r.get('success_rate', 0))
                rcp = float(r.get('avg_rcp', 0))
            except (TypeError, ValueError):
                continue
            rows.append((sigma, ws, sr, rcp))
    return rows


def main(args):
    w_rows = load_pareto(args.pareto_wbit)
    b_rows = load_pareto(args.pareto_binary)
    a_rows = load_pareto(args.pareto_adaptive)

    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    if w_rows:
        plt.scatter([r[3] for r in w_rows], [r[2] for r in w_rows], marker='o', label='wbit')
    if b_rows:
        plt.scatter([r[3] for r in b_rows], [r[2] for r in b_rows], marker='x', label='binary')
    if a_rows:
        plt.scatter([r[3] for r in a_rows], [r[2] for r in a_rows], marker='s', label='adaptive')
    plt.xlabel('Avg RCP')
    plt.ylabel('Success Rate')
    plt.title('Phase 2 Pareto Frontier (Success vs RCP)')
    plt.grid(True)
    if w_rows or b_rows:
        plt.legend()
    else:
        plt.title('Phase 2 Pareto Frontier (no data)')
    out_path = os.path.join(args.output_dir, 'pareto_success_vs_rcp.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Pareto plot written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pareto_wbit', type=str, default='results/phase2/phase2_pareto_wbit.csv')
    parser.add_argument('--pareto_binary', type=str, default='results/phase2/phase2_pareto_binary.csv')
    parser.add_argument('--pareto_adaptive', type=str, default='results/phase2/phase2_pareto_adaptive.csv')
    parser.add_argument('--output_dir', type=str, default='results/phase2/plots')
    args = parser.parse_args()
    main(args)
