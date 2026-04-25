import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def generate_report(parquet_path):
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Ensure index is datetime for plotting
    if 'raw.timestamp_ns' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['raw.timestamp_ns'], unit='ns')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('W-bit Level 3 Bench Master Log', fontsize=16)

    # 1. Tradeoff Vector (Fidelity & Mode)
    ax1 = axes[0]
    ax1.plot(df.index, df['raw.fidelity_score'], label='Fidelity %', color='blue', alpha=0.8)
    ax1.set_ylabel('Fidelity %', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(99.0, color='blue', linestyle='--', alpha=0.3, label='99% Target')
    ax1.axhline(95.0, color='red', linestyle='--', alpha=0.3, label='95% Critical')
    
    ax1_twin = ax1.twinx()
    ax1_twin.step(df.index, df['raw.os_mode_state'], label='OS Mode (R_eff)', color='purple', where='post')
    ax1_twin.set_ylabel('R_eff Mode', color='purple')
    ax1_twin.set_yticks([3, 5, 7, 9])
    ax1_twin.tick_params(axis='y', labelcolor='purple')
    
    # Highlight events
    if 'raw.event_marker' in df.columns:
        for t, marker in df[df['raw.event_marker'].notna()]['raw.event_marker'].items():
            if marker == 'SHOCK_INJECTED':
                ax1.axvline(t, color='red', linestyle='-', alpha=0.5)
            elif marker == 'DISTURBANCE_ON':
                ax1.axvline(t, color='orange', linestyle='-', alpha=0.5)

    # 2. Thermal Profile & Self-Load
    ax2 = axes[1]
    ax2.plot(df.index, df['raw.temp_center_c'], label='Temp Center', color='darkred')
    ax2.plot(df.index, df['raw.temp_edge_c'], label='Temp Edge', color='salmon', alpha=0.7)
    ax2.plot(df.index, df['raw.temp_far_c'], label='Temp Far', color='orange', alpha=0.7)
    if 'derived.thermal_self_load' in df.columns:
        ax2.plot(df.index, df['derived.thermal_self_load'], label='Self-Heating Load', color='purple', linestyle='--')
    ax2.set_ylabel('Temperature (C)')
    ax2.legend(loc='upper right')

    # 3. Estimator Health
    ax3 = axes[2]
    ax3.plot(df.index, df['raw.n_est_slow'], label='Sentinel N_est', color='green')
    if 'raw.true_noise_c63' in df.columns:
        ax3.plot(df.index, df['raw.true_noise_c63'], label='True Noise (C63)', color='black', alpha=0.5, linestyle=':')
    ax3.set_ylabel('Noise (mV)')
    ax3.legend(loc='upper right')
    
    if 'derived.n_est_error' in df.columns:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df.index, df['derived.n_est_error'], label='Estimator Error %', color='gray', alpha=0.5)
        ax3_twin.set_ylabel('Error %', color='gray')
        ax3_twin.axhline(-15.0, color='red', linestyle='--', alpha=0.3)

    # 4. Power & Energy
    ax4 = axes[3]
    ax4.plot(df.index, df['raw.energy_cost_est'], label='Energy (fJ/op)', color='brown')
    ax4.axhline(1000.0, color='red', linestyle='--', alpha=0.3, label='Budget Limit')
    ax4.set_ylabel('fJ / op')
    ax4.set_xlabel('Time')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    output_png = parquet_path.replace('.parquet', '_report.png')
    plt.savefig(output_png, dpi=300)
    print(f"Generated visual report: {output_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python level3_plot_report.py <path_to_analyzed_parquet>")
        sys.exit(1)
        
    parquet_path = sys.argv[1]
    if not os.path.exists(parquet_path):
        print(f"Error: File {parquet_path} not found.")
        sys.exit(1)
        
    generate_report(parquet_path)
