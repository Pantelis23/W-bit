import argparse
import csv
import os
from collections import defaultdict


def load_summary(path):
    if not os.path.exists(path):
        print(f"Missing summary: {path}")
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_lookup(rows):
    lookup = defaultdict(list)  # sigma -> list of (weight_scale, success_rate, avg_rcp)
    sigmas = set()
    for r in rows:
        try:
            sigma = float(r.get('sigma', 0))
            ws = float(r.get('weight_scale', 0))
            success_rate = float(r.get('success_rate', 0))
            avg_rcp = float(r.get('avg_rcp', 0))
        except (TypeError, ValueError):
            continue
        sigmas.add(sigma)
        lookup[sigma].append((ws, success_rate, avg_rcp))
    return lookup, sigmas


def choose_frontier(lookup, sigma, target_success):
    entries = lookup.get(sigma, [])
    if not entries:
        return None
    meets_target_list = [(rcp, ws, sr) for ws, sr, rcp in entries if sr >= target_success]
    if meets_target_list:
        meets_target_list.sort()  # min rcp, then ws
        best_rcp, best_ws, best_sr = meets_target_list[0]
        return best_ws, best_sr, best_rcp, 1
    # fallback: max success_rate, then min rcp, then min ws
    entries_sorted = sorted(entries, key=lambda x: (-x[1], x[2], x[0]))
    best_ws, best_sr, best_rcp = entries_sorted[0]
    return best_ws, best_sr, best_rcp, 0


def main(args):
    w_rows = load_summary(args.wbit_summary)
    b_rows = load_summary(args.binary_summary)
    a_rows = load_summary(args.adaptive_summary) if args.adaptive_summary else []

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not w_rows or not b_rows:
        with open(args.out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sigma',
                'target_success',
                'weight_scale_wbit',
                'success_rate_wbit',
                'avg_rcp_wbit',
                'meets_target_wbit',
                'weight_scale_binary',
                'success_rate_binary',
                'avg_rcp_binary',
                'meets_target_binary',
                'weight_scale_adaptive',
                'success_rate_adaptive',
                'avg_rcp_adaptive',
                'meets_target_adaptive',
                'delta_avg_rcp',
                'delta_avg_rcp_adaptive'
            ])
        print(f"Wrote frontier summary to {args.out}")
        return

    w_lookup, w_sigmas = build_lookup(w_rows)
    b_lookup, b_sigmas = build_lookup(b_rows)
    a_lookup, a_sigmas = build_lookup(a_rows) if a_rows else ({}, set())

    common_sigmas = set(w_sigmas).intersection(b_sigmas)
    if a_rows:
        common_sigmas = common_sigmas.intersection(a_sigmas)
    common_sigmas = sorted(common_sigmas)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = [
        'sigma',
        'target_success',
        'weight_scale_wbit',
        'success_rate_wbit',
        'avg_rcp_wbit',
        'meets_target_wbit',
        'weight_scale_binary',
        'success_rate_binary',
        'avg_rcp_binary',
        'meets_target_binary',
        'weight_scale_adaptive',
        'success_rate_adaptive',
        'avg_rcp_adaptive',
        'meets_target_adaptive',
        'delta_avg_rcp',
        'delta_avg_rcp_adaptive'
    ]

    rows_out = []
    for sigma in common_sigmas:
        w_best = choose_frontier(w_lookup, sigma, args.target_success)
        b_best = choose_frontier(b_lookup, sigma, args.target_success)
        a_best = choose_frontier(a_lookup, sigma, args.target_success) if a_rows else None
        if not w_best or not b_best:
            continue
        ws_w, sr_w, rcp_w, meet_w = w_best
        ws_b, sr_b, rcp_b, meet_b = b_best
        entry = {
            'sigma': sigma,
            'target_success': args.target_success,
            'weight_scale_wbit': ws_w,
            'success_rate_wbit': sr_w,
            'avg_rcp_wbit': rcp_w,
            'meets_target_wbit': meet_w,
            'weight_scale_binary': ws_b,
            'success_rate_binary': sr_b,
            'avg_rcp_binary': rcp_b,
            'meets_target_binary': meet_b,
            'weight_scale_adaptive': '',
            'success_rate_adaptive': '',
            'avg_rcp_adaptive': '',
            'meets_target_adaptive': '',
            'delta_avg_rcp': rcp_w - rcp_b,
            'delta_avg_rcp_adaptive': ''
        }
        if a_best:
            ws_a, sr_a, rcp_a, meet_a = a_best
            entry.update({
                'weight_scale_adaptive': ws_a,
                'success_rate_adaptive': sr_a,
                'avg_rcp_adaptive': rcp_a,
                'meets_target_adaptive': meet_a,
                'delta_avg_rcp_adaptive': rcp_w - rcp_a
            })
        rows_out.append(entry)

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote frontier summary to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wbit_summary', type=str, default='results/phase2/wbit/expB_grid/summary.csv')
    parser.add_argument('--binary_summary', type=str, default='results/phase2/binary/expB_grid/summary.csv')
    parser.add_argument('--adaptive_summary', type=str, default='results/phase2/adaptive/expB_grid/summary.csv')
    parser.add_argument('--out', type=str, default='results/phase2/phase2_frontier_summary.csv')
    parser.add_argument('--target_success', type=float, default=0.9)
    args = parser.parse_args()
    main(args)
