import os
import csv
import glob
import argparse
import matplotlib.pyplot as plt

def plot_exp_a(results_dir, output_dir):
    print(f"Loading CSVs from {results_dir}...")
    
    # Recursive search for results.csv
    csv_files = glob.glob(os.path.join(results_dir, '**/results.csv'), recursive=True)
    
    if not csv_files:
        print("No results.csv files found.")
        return
        
    rows = []
    for f in csv_files:
        try:
            # Only read if not empty
            if os.path.getsize(f) > 0:
                with open(f, newline='') as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        rows.append(r)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not rows:
        print("No valid data loaded.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect unique layouts and grids
    layouts = sorted({r.get('layout') for r in rows if r.get('layout') is not None})
    
    for layout in layouts:
        print(f"Plotting for layout: {layout}")
        layout_rows = [r for r in rows if r.get('layout') == layout]
        grids = sorted({g for g in (r.get('grid') for r in layout_rows) if g is not None})
        
        # 1. Feasible Rate vs Density
        plt.figure(figsize=(10, 6))
        for g in grids:
            df_g = [r for r in layout_rows if r.get('grid') == g]
            # Group by density
            grouped = {}
            for r in df_g:
                try:
                    dens = float(r.get('obstacle_density', 0))
                    feas = float(r.get('feasible', 0))
                except (TypeError, ValueError):
                    continue
                grouped.setdefault(dens, []).append(feas)
            densities = sorted(grouped.keys())
            if densities:
                mean_vals = [sum(grouped[d])/len(grouped[d]) for d in densities]
                plt.plot(densities, mean_vals, marker='o', label=f'Grid {g}x{g}')
        
        plt.title(f'Feasibility Rate vs Obstacle Density ({layout})')
        plt.xlabel('Density')
        plt.ylabel('Feasible Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{layout}_feasibility.png'))
        plt.close()
        
        # 2. Avg RCP (Success Only) vs Density
        plt.figure(figsize=(10, 6))
        for g in grids:
            df_g = [r for r in layout_rows if r.get('grid') == g]
            # Filter success = 1
            grouped = {}
            for r in df_g:
                try:
                    success = float(r.get('success', 0))
                    if success != 1:
                        continue
                    dens = float(r.get('obstacle_density', 0))
                    rcp = float(r.get('rcp', 0))
                except (TypeError, ValueError):
                    continue
                grouped.setdefault(dens, []).append(rcp)
            densities = sorted(grouped.keys())
            if densities:
                mean_vals = [sum(grouped[d])/len(grouped[d]) for d in densities]
                plt.plot(densities, mean_vals, marker='o', label=f'Grid {g}x{g}')
        
        plt.title(f'Avg RCP (Success Only) vs Density ({layout})')
        plt.xlabel('Density')
        plt.ylabel('RCP')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{layout}_rcp.png'))
        plt.close()
        
        # 3. Mean Detour Ratio (Success Only) vs Density
        plt.figure(figsize=(10, 6))
        for g in grids:
            df_g = [r for r in layout_rows if r.get('grid') == g]
            grouped = {}
            for r in df_g:
                try:
                    success = float(r.get('success', 0))
                    if success != 1:
                        continue
                    dens = float(r.get('obstacle_density', 0))
                    detour = float(r.get('detour_ratio', 0))
                except (TypeError, ValueError):
                    continue
                grouped.setdefault(dens, []).append(detour)
            densities = sorted(grouped.keys())
            if densities:
                mean_vals = [sum(grouped[d])/len(grouped[d]) for d in densities]
                plt.plot(densities, mean_vals, marker='o', label=f'Grid {g}x{g}')
                
        plt.title(f'Mean Detour Ratio (Success Only) vs Density ({layout})')
        plt.xlabel('Density')
        plt.ylabel('Detour Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{layout}_detour.png'))
        plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/expA')
    parser.add_argument('--output_dir', type=str, default='results/expA/plots')
    args = parser.parse_args()
    
    # Needs pandas and matplotlib.
    # In this environment, we might not have them.
    # We'll wrap in try-except block in main execution or just assume they exist per instructions "data-integrity fix" implied analysis capabilities?
    # Instructions said "Add plotting utility". Assuming libraries available.
    try:
        plot_exp_a(args.results_dir, args.output_dir)
    except ImportError:
        print("pandas or matplotlib not installed. Skipping plotting.")
