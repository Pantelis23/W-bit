import argparse
import csv
import os

import matplotlib.pyplot as plt


def read_rows(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(row, key, default=0.0):
    try:
        return float(row[key])
    except Exception:
        return default


def group_by(rows, key):
    groups = {}
    for row in rows:
        k = row.get(key)
        groups.setdefault(k, []).append(row)
    return groups


def plot_loss_vs_R(rows, out_dir):
    by_sigma = group_by(rows, 'sigma')
    for sigma, sigma_rows in by_sigma.items():
        plt.figure()
        for mode in sorted(set(r.get('mode') for r in sigma_rows)):
            sub = [r for r in sigma_rows if r.get('mode') == mode]
            sub = sorted(sub, key=lambda r: to_float(r, 'R_effective'))
            xs = [to_float(r, 'R_effective') for r in sub]
            ys = [to_float(r, 'loss_delta') for r in sub]
            plt.plot(xs, ys, marker='o', label=f"{mode}")
        plt.xlabel("R_effective")
        plt.ylabel("loss_delta vs baseline logits")
        plt.title(f"Loss delta vs R (sigma={sigma})")
        plt.legend()
        plt.grid(True)
        outfile = os.path.join(out_dir, f"real_layer_loss_vs_R_sigma{sigma}.png")
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()


def plot_success_vs_sigma(rows, out_dir):
    by_R = group_by(rows, 'R_effective')
    for R_eff, R_rows in by_R.items():
        plt.figure()
        for mode in sorted(set(r.get('mode') for r in R_rows)):
            sub = [r for r in R_rows if r.get('mode') == mode]
            sub = sorted(sub, key=lambda r: to_float(r, 'sigma'))
            xs = [to_float(r, 'sigma') for r in sub]
            ys = [to_float(r, 'success_rate') for r in sub]
            plt.plot(xs, ys, marker='o', label=f"{mode}")
        plt.xlabel("sigma (weight noise)")
        plt.ylabel("success_rate (pred match to baseline)")
        plt.title(f"Robustness vs sigma (R_effective={R_eff})")
        plt.legend()
        plt.grid(True)
        outfile = os.path.join(out_dir, f"real_layer_success_vs_sigma_R{R_eff}.png")
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot loss_delta vs R and robustness vs sigma from real_layer_quant CSV.")
    parser.add_argument('--input_csv', type=str, default='results/phase2/real_layer_quant.csv')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output dir (default: same as CSV)')
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    if not rows:
        print("No rows to plot.")
        return

    out_dir = args.output_dir if args.output_dir else os.path.dirname(args.input_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_loss_vs_R(rows, out_dir)
    plot_success_vs_sigma(rows, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
