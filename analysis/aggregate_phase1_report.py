import csv
import os
import glob
import statistics
import argparse

def aggregate_reports(args):
    print("=== Phase 1 Report Aggregation ===")
    
    results_data = []
    
    # Experiment A
    exp_a_path = os.path.join(args.expA_dir, 'summary.csv')
    if not os.path.exists(exp_a_path):
        # Try recursive search
        csvs = glob.glob(os.path.join(args.expA_dir, '**', 'summary.csv'), recursive=True)
        if csvs:
            print(f"Found {len(csvs)} Exp A summary files.")
            for f in csvs:
                with open(f, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        row['experiment'] = 'A'
                        results_data.append(row)
        else:
            print("Warning: No Exp A summary.csv found.")
    else:
        with open(exp_a_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['experiment'] = 'A'
                results_data.append(row)
                
    # Experiment B Grid
    exp_b_grid_path = os.path.join(args.expB_dir, 'summary.csv')
    if os.path.exists(exp_b_grid_path):
        print(f"Found Exp B Grid summary.")
        with open(exp_b_grid_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['experiment'] = 'B'
                results_data.append(row)
    else:
        print("Warning: No Exp B Grid summary.csv found.")

    # Experiment B Noise Breakdown (optional)
    exp_b_noise_path = os.path.join(args.expB_noise_dir, 'results.csv')
    if os.path.exists(exp_b_noise_path):
        print("Found Exp B Noise Breakdown results.")
        with open(exp_b_noise_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['experiment'] = 'B_noise'
                results_data.append(row)
    else:
        print("Info: No Exp B Noise results.csv found.")
        
    # Experiment C
    exp_c_path = os.path.join(args.expC_dir, 'results.csv')
    if os.path.exists(exp_c_path):
        print(f"Found Exp C results.")
        with open(exp_c_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Aggregate C manually since it's raw trials
            # Group by H, allow_direct_ab_to_y
            grouped = {}
            for r in rows:
                key = (r['H'], r.get('allow_direct_ab_to_y', 'True'))
                if key not in grouped:
                    grouped[key] = {'success': [], 'mse': [], 'acc': [], 'rcp': []}
                
                grouped[key]['success'].append(int(r['success']))
                # C1: best_search_mse in newer runs
                if 'best_search_mse' in r:
                    val = r['best_search_mse']
                elif 'best_mse' in r:
                    val = r['best_mse']
                else:
                    val = 0.0 # Default fallback
                grouped[key]['mse'].append(float(val) if val else 0.0)
                grouped[key]['acc'].append(float(r.get('best_acc', 0.0)))
                grouped[key]['rcp'].append(float(r['inference_rcp']))
                
            for (h, direct), vals in grouped.items():
                row = {
                    'experiment': 'C',
                    'H': h,
                    'direct_conn': direct,
                    'success_rate': statistics.mean(vals['success']),
                    'mean_best_mse': statistics.mean(vals['mse']),
                    'mean_best_acc': statistics.mean(vals['acc']),
                    'avg_rcp': statistics.mean(vals['rcp']),
                    'trials': len(vals['success'])
                }
                results_data.append(row)
    else:
        print("Warning: No Exp C results.csv found.")
        
    # Write Consolidated Report
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_path = args.out
    
    # Collect all possible field names
    fieldnames = set()
    for r in results_data:
        fieldnames.update(r.keys())
    
    # Prioritize key fields order
    priority = ['experiment', 'layout', 'grid', 'obstacle_density', 'sigma', 'weight_scale', 'H', 'direct_conn', 'success_rate', 'avg_rcp', 'avg_rcp_success']
    ordered_fields = [f for f in priority if f in fieldnames] + sorted([f for f in fieldnames if f not in priority])
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(results_data)
        
    print(f"Report written to {out_path}")
    print(f"Total rows: {len(results_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expA_dir', type=str, default='results/expA')
    parser.add_argument('--expB_dir', type=str, default='results/expB_grid')
    parser.add_argument('--expB_noise_dir', type=str, default='results/expB')
    parser.add_argument('--expC_dir', type=str, default='results/expC')
    parser.add_argument('--out', type=str, default='results/phase1_report.csv')
    args = parser.parse_args()
    aggregate_reports(args)
