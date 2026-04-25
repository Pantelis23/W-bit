import argparse
import csv
import glob
import os

def collect_mode(mode_label, expA_dir, expB_dir, expB_noise_dir, expC_dir):
    rows = []

    # Experiment A
    exp_a_path = os.path.join(expA_dir, 'summary.csv')
    if os.path.exists(exp_a_path):
        with open(exp_a_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k: v for k, v in row.items() if k}
                row['mode'] = mode_label
                row['experiment'] = row.get('experiment', 'A')
                row['summary_type'] = 'raw'
                rows.append(row)
    else:
        csvs = glob.glob(os.path.join(expA_dir, '**', 'summary.csv'), recursive=True)
        for fpath in csvs:
            with open(fpath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = {k: v for k, v in row.items() if k}
                    row['mode'] = mode_label
                    row['experiment'] = row.get('experiment', 'A')
                    row['summary_type'] = 'raw'
                    rows.append(row)

    # Experiment B Grid
    exp_b_grid_path = os.path.join(expB_dir, 'summary.csv')
    if os.path.exists(exp_b_grid_path):
        with open(exp_b_grid_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k: v for k, v in row.items() if k}
                row['mode'] = mode_label
                row['experiment'] = row.get('experiment', 'B_grid')
                row['summary_type'] = 'raw'
                rows.append(row)

    # Experiment B Noise
    exp_b_noise_path = os.path.join(expB_noise_dir, 'results.csv')
    if os.path.exists(exp_b_noise_path):
        with open(exp_b_noise_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k: v for k, v in row.items() if k}
                row['mode'] = mode_label
                row['experiment'] = row.get('experiment', 'B_noise')
                row['summary_type'] = 'raw'
                rows.append(row)

    # Experiment C
    exp_c_path = os.path.join(expC_dir, 'results.csv')
    if os.path.exists(exp_c_path):
        with open(exp_c_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k: v for k, v in row.items() if k}
                row['mode'] = mode_label
                row['experiment'] = row.get('experiment', 'C')
                row['summary_type'] = 'raw'
                rows.append(row)

    return rows

def aggregate_phase2(args):
    print("=== Phase 2 Aggregation (wbit vs binary scaffold) ===")
    results = []

    results.extend(collect_mode('wbit', args.wbit_expA_dir, args.wbit_expB_dir, args.wbit_expB_noise_dir, args.wbit_expC_dir))
    results.extend(collect_mode('binary', args.binary_expA_dir, args.binary_expB_dir, args.binary_expB_noise_dir, args.binary_expC_dir))
    if args.adaptive_expA_dir or args.adaptive_expB_dir or args.adaptive_expB_noise_dir or args.adaptive_expC_dir:
        results.extend(collect_mode('adaptive', args.adaptive_expA_dir, args.adaptive_expB_dir, args.adaptive_expB_noise_dir, args.adaptive_expC_dir))

    summary_rows = []

    # Summary: Exp B Noise
    b_noise_rows = [r for r in results if r.get('experiment') == 'B_noise']
    if b_noise_rows:
        grouped = {}
        for r in b_noise_rows:
            key = (r.get('mode'), r.get('sigma'), r.get('weight_scale'))
            grouped.setdefault(key, []).append(r)
        for (mode, sigma, weight_scale), rows in grouped.items():
            successes = [int(r.get('success', 0)) for r in rows if r.get('success') is not None]
            steps_vals = [float(r.get('steps', 0)) for r in rows if r.get('steps') is not None]
            rcp_vals = [float(r.get('rcp', 0)) for r in rows if r.get('rcp') is not None]
            conf_vals = [float(r.get('final_confidence', 0)) for r in rows if r.get('final_confidence') is not None]
            trials = len(rows)
            success_rate = sum(successes) / trials if trials else 0.0
            summary_rows.append({
                'mode': mode,
                'experiment': 'B_noise_summary',
                'sigma': sigma,
                'weight_scale': weight_scale,
                'success_rate': success_rate,
                'mean_steps': sum(steps_vals) / trials if trials else 0.0,
                'mean_rcp': sum(rcp_vals) / trials if trials else 0.0,
                'mean_final_conf': sum(conf_vals) / trials if trials else 0.0,
                'trials': trials,
                'summary_type': 'summary_B_noise'
            })

    # Summary: Exp C by H
    c_rows = [r for r in results if r.get('experiment') == 'C']
    if c_rows:
        grouped = {}
        for r in c_rows:
            key = (r.get('mode'), r.get('H'))
            grouped.setdefault(key, []).append(r)
        for (mode, H), rows in grouped.items():
            successes = [int(r.get('success', 0)) for r in rows if r.get('success') is not None]
            trials = len(rows)
            summary_rows.append({
                'mode': mode,
                'experiment': 'C_summary',
                'H': H,
                'success_rate': sum(successes) / trials if trials else 0.0,
                'trials': trials,
                'summary_type': 'summary_C'
            })

    if results:
        fieldnames = set()
        for r in results + summary_rows:
            fieldnames.update(r.keys())
        fieldnames = {f for f in fieldnames if f is not None}
        ordered = ['summary_type', 'mode', 'experiment', 'layout', 'grid', 'obstacle_density', 'sigma', 'weight_scale', 'H', 'direct_conn', 'success_rate', 'mean_steps', 'mean_rcp', 'mean_final_conf', 'avg_rcp_success']
        ordered_fields = [f for f in ordered if f in fieldnames] + [f for f in sorted(fieldnames) if f not in ordered]
    else:
        ordered_fields = ['mode']

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(results + summary_rows)

    print(f"Wrote Phase 2 scaffold report to {args.out}")
    print(f"Total rows: {len(results) + len(summary_rows)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wbit_expA_dir', type=str, default='results/phase2/wbit/expA')
    parser.add_argument('--wbit_expB_dir', type=str, default='results/phase2/wbit/expB_grid')
    parser.add_argument('--wbit_expB_noise_dir', type=str, default='results/phase2/wbit/expB')
    parser.add_argument('--wbit_expC_dir', type=str, default='results/phase2/wbit/expC')

    parser.add_argument('--binary_expA_dir', type=str, default='results/phase2/binary/expA')
    parser.add_argument('--binary_expB_dir', type=str, default='results/phase2/binary/expB_grid')
    parser.add_argument('--binary_expB_noise_dir', type=str, default='results/phase2/binary/expB')
    parser.add_argument('--binary_expC_dir', type=str, default='results/phase2/binary/expC')
    parser.add_argument('--adaptive_expA_dir', type=str, default='results/phase2/adaptive/expA')
    parser.add_argument('--adaptive_expB_dir', type=str, default='results/phase2/adaptive/expB_grid')
    parser.add_argument('--adaptive_expB_noise_dir', type=str, default='results/phase2/adaptive/expB')
    parser.add_argument('--adaptive_expC_dir', type=str, default='results/phase2/adaptive/expC')

    parser.add_argument('--out', type=str, default='results/phase2/phase2_report.csv')

    args = parser.parse_args()
    aggregate_phase2(args)
