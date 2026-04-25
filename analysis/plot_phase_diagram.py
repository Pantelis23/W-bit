import argparse
import csv
import os

import matplotlib.pyplot as plt


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def unique_sorted(values, cast_fn):
    return sorted({cast_fn(v) for v in values})


def build_grid(rows, weight_scale, mode):
    filtered = [r for r in rows if r.get('weight_scale') == weight_scale and r.get('mode') == mode]
    sigmas = unique_sorted([r['sigma'] for r in filtered], float)
    Rs = unique_sorted([r['R_effective'] for r in filtered], float)
    sigma_index = {s: idx for idx, s in enumerate(sigmas)}
    R_index = {r: idx for idx, r in enumerate(Rs)}

    success_grid = [[None for _ in sigmas] for _ in Rs]
    phase_grid = [[None for _ in sigmas] for _ in Rs]

    for r in filtered:
        s_val = float(r['sigma'])
        r_val = float(r['R_effective'])
        sr = float(r['success_rate'])
        phase = r.get('phase_label', 'unknown')
        success_grid[R_index[r_val]][sigma_index[s_val]] = sr
        phase_grid[R_index[r_val]][sigma_index[s_val]] = phase

    return sigmas, Rs, success_grid, phase_grid


def render_heatmap(sigmas, Rs, grid, title, outfile, cmap='viridis', phase=False):
    plt.figure(figsize=(8, 4.5))
    if phase:
        phase_map = {'fail': 0, 'edge': 1, 'good': 2, None: -1, 'unknown': -1}
        numeric_grid = [[phase_map.get(cell, -1) for cell in row] for row in grid]
        im = plt.imshow(numeric_grid, origin='lower', aspect='auto', cmap=plt.cm.get_cmap('coolwarm', 3))
        cbar = plt.colorbar(im, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['fail', 'edge', 'good'])
    else:
        im = plt.imshow(grid, origin='lower', aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)
        plt.colorbar(im, label='success_rate')
    plt.xticks(range(len(sigmas)), [str(s) for s in sigmas], rotation=45)
    plt.yticks(range(len(Rs)), [str(r) for r in Rs])
    plt.xlabel("sigma")
    plt.ylabel("R_effective")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot phase_diagram.csv heatmaps (success_rate and phase labels).")
    parser.add_argument('--input_csv', type=str, default='results/phase2/wbit/expB_grid/phase_diagram.csv')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output dir (default: CSV dir)')
    parser.add_argument('--mode', type=str, default='wbit', help='Mode to filter (wbit/binary/adaptive)')
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    if not rows:
        print("No rows found.")
        return

    out_dir = args.output_dir if args.output_dir else os.path.dirname(args.input_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    weight_scales = sorted({r['weight_scale'] for r in rows})
    for ws in weight_scales:
        sigmas, Rs, success_grid, phase_grid = build_grid(rows, ws, args.mode)
        if not sigmas or not Rs:
            continue
        success_out = os.path.join(out_dir, f"phase_success_ws{ws}_{args.mode}.png")
        phase_out = os.path.join(out_dir, f"phase_labels_ws{ws}_{args.mode}.png")
        render_heatmap(sigmas, Rs, success_grid, f"Success Rate (ws={ws}, mode={args.mode})", success_out)
        render_heatmap(sigmas, Rs, phase_grid, f"Phase Labels (ws={ws}, mode={args.mode})", phase_out, phase=True)
    print(f"Phase plots written to {out_dir}")


if __name__ == "__main__":
    main()
