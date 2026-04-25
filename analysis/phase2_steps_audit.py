import argparse
import csv
import os
import statistics


def load_results(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def summarize(rows):
    steps_all = []
    steps_success = []
    for r in rows:
        if 'feasible' in r and r.get('feasible') not in (None, '', '1', 1, True):
            # Skip infeasible maps for all metrics
            continue
        try:
            s = float(r.get('steps', 0))
        except (TypeError, ValueError):
            continue
        steps_all.append(s)
        if r.get('success') in ('1', 1, True, 'True'):
            steps_success.append(s)

    mean_steps = statistics.mean(steps_success) if steps_success else 0
    median_steps = statistics.median(steps_success) if steps_success else 0
    zero_all = (sum(1 for s in steps_all if s == 0) / len(steps_all)) if steps_all else 0
    zero_success = (sum(1 for s in steps_success if s == 0) / len(steps_success)) if steps_success else 0
    return mean_steps, median_steps, zero_all, zero_success


def main(args):
    entries = []
    means = {}
    modes = []
    phase2_root = os.path.join('results', 'phase2')
    if os.path.isdir(phase2_root):
        for name in os.listdir(phase2_root):
            if name.startswith('.'):
                continue
            mode_path = os.path.join(phase2_root, name)
            if os.path.isdir(mode_path):
                modes.append(name)
    if not modes:
        modes = ['wbit', 'binary']

    for mode in sorted(modes):
        expA_path = os.path.join(phase2_root, mode, 'expA', 'results.csv')
        expB_path = os.path.join(phase2_root, mode, 'expB', 'results.csv')

        mean_a, median_a, zero_a_all, zero_a_success = summarize(load_results(expA_path))
        mean_b, median_b, zero_b_all, zero_b_success = summarize(load_results(expB_path))

        entries.append([mode, 'expA', mean_a, median_a, zero_a_all, zero_a_success])
        entries.append([mode, 'expB', mean_b, median_b, zero_b_all, zero_b_success])
        means[(mode, 'expA')] = mean_a
        means[(mode, 'expB')] = mean_b

    # Ratios to wbit
    for exp in ['expA', 'expB']:
        w_mean = means.get(('wbit', exp))
        for mode in modes:
            if mode == 'wbit':
                continue
            b_mean = means.get((mode, exp))
            if w_mean and b_mean is not None:
                ratio = b_mean / w_mean if w_mean != 0 else 0.0
                label = f"{mode}_to_wbit_ratio"
                entries.append([label, exp, ratio, '', '', ''])

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'experiment', 'mean_steps', 'median_steps', 'zero_step_fraction_all', 'zero_step_fraction_success'])
        writer.writerows(entries)

    print(f"Wrote steps audit to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='results/phase2/phase2_steps_audit.csv')
    args = parser.parse_args()
    main(args)
