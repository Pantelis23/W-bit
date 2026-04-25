import argparse
import csv
import os


def load_summary(path):
    rows = []
    if not os.path.exists(path):
        print(f"Missing summary: {path}")
        return rows
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def pareto_for_mode(rows):
    by_sigma = {}
    for r in rows:
        try:
            sigma = float(r.get('sigma', 0))
            ws = float(r.get('weight_scale', 0))
            sr = float(r.get('success_rate', 0))
            rcp = float(r.get('avg_rcp', 0))
        except (TypeError, ValueError):
            continue
        by_sigma.setdefault(sigma, []).append((ws, sr, rcp))

    pareto_rows = []
    for sigma, points in by_sigma.items():
        for ws, sr, rcp in points:
            dominated = False
            for ws2, sr2, rcp2 in points:
                if sr2 >= sr and rcp2 <= rcp and ((sr2 > sr) or (rcp2 < rcp)):
                    dominated = True
                    break
            if not dominated:
                pareto_rows.append((sigma, ws, sr, rcp))
    pareto_rows.sort(key=lambda x: (x[0], -x[2], x[3], x[1]))
    return pareto_rows


def write_pareto(rows, mode, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    header = ['mode', 'sigma', 'weight_scale', 'success_rate', 'avg_rcp', 'is_pareto']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sigma, ws, sr, rcp in rows:
            writer.writerow([mode, sigma, ws, sr, rcp, 1])


def main(args):
    w_rows = load_summary(args.wbit_summary)
    b_rows = load_summary(args.binary_summary)
    a_rows = load_summary(args.adaptive_summary) if args.adaptive_summary else []

    pareto_w = pareto_for_mode(w_rows)
    pareto_b = pareto_for_mode(b_rows)
    pareto_a = pareto_for_mode(a_rows) if a_rows else []

    write_pareto(pareto_w, 'wbit', args.out_wbit)
    write_pareto(pareto_b, 'binary', args.out_binary)
    if a_rows:
        write_pareto(pareto_a, 'adaptive', args.out_adaptive)

    if a_rows:
        print(f"Wrote Pareto CSVs to {args.out_wbit}, {args.out_binary}, {args.out_adaptive}")
    else:
        print(f"Wrote Pareto CSVs to {args.out_wbit} and {args.out_binary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wbit_summary', type=str, default='results/phase2/wbit/expB_grid/summary.csv')
    parser.add_argument('--binary_summary', type=str, default='results/phase2/binary/expB_grid/summary.csv')
    parser.add_argument('--out_wbit', type=str, default='results/phase2/phase2_pareto_wbit.csv')
    parser.add_argument('--out_binary', type=str, default='results/phase2/phase2_pareto_binary.csv')
    parser.add_argument('--adaptive_summary', type=str, default='results/phase2/adaptive/expB_grid/summary.csv')
    parser.add_argument('--out_adaptive', type=str, default='results/phase2/phase2_pareto_adaptive.csv')
    args = parser.parse_args()
    main(args)
