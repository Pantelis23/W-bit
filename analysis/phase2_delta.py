import argparse
import csv
import os

def load_summary(path):
    rows = []
    if not os.path.exists(path):
        print(f"Missing report: {path}")
        return rows
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main(args):
    rows = load_summary(args.report)
    b_rows = [r for r in rows if r.get('summary_type') == 'summary_B_noise']
    if not b_rows:
        print("No summary_B_noise rows found.")
    grouped = {}
    for r in b_rows:
        try:
            sigma = float(r.get('sigma', 0))
            weight_scale = float(r.get('weight_scale', 0))
            mode = r.get('mode')
            success_rate = float(r.get('success_rate', 0))
            mean_rcp = float(r.get('mean_rcp', 0))
        except (TypeError, ValueError):
            continue
        key = (sigma, weight_scale, mode)
        grouped[key] = {'success_rate': success_rate, 'mean_rcp': mean_rcp}

    sigmas = sorted({k[0] for k in grouped.keys()})
    weight_scales = sorted({k[1] for k in grouped.keys()})

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w', newline='') as f:
        fieldnames = [
            'sigma', 'weight_scale',
            'success_rate_wbit', 'success_rate_binary', 'success_rate_adaptive',
            'delta_success_rate', 'delta_success_rate_adaptive',
            'mean_rcp_wbit', 'mean_rcp_binary', 'mean_rcp_adaptive',
            'delta_mean_rcp', 'delta_mean_rcp_adaptive'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sigma in sigmas:
            for ws in weight_scales:
                wbit = grouped.get((sigma, ws, 'wbit'))
                binary = grouped.get((sigma, ws, 'binary'))
                adaptive = grouped.get((sigma, ws, 'adaptive'))
                if not wbit:
                    continue
                row = {
                    'sigma': sigma,
                    'weight_scale': ws,
                    'success_rate_wbit': wbit.get('success_rate'),
                    'success_rate_binary': binary['success_rate'] if binary else None,
                    'success_rate_adaptive': adaptive['success_rate'] if adaptive else None,
                    'delta_success_rate': (wbit['success_rate'] - binary['success_rate']) if binary else None,
                    'delta_success_rate_adaptive': (wbit['success_rate'] - adaptive['success_rate']) if adaptive else None,
                    'mean_rcp_wbit': wbit.get('mean_rcp'),
                    'mean_rcp_binary': binary['mean_rcp'] if binary else None,
                    'mean_rcp_adaptive': adaptive['mean_rcp'] if adaptive else None,
                    'delta_mean_rcp': (wbit['mean_rcp'] - binary['mean_rcp']) if binary else None,
                    'delta_mean_rcp_adaptive': (wbit['mean_rcp'] - adaptive['mean_rcp']) if adaptive else None
                }
                writer.writerow(row)

    print(f"Wrote delta summary to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, default='results/phase2/phase2_report.csv')
    parser.add_argument('--out', type=str, default='results/phase2/phase2_delta_summary.csv')
    args = parser.parse_args()
    main(args)
