import pandas as pd
import numpy as np
import sys
import os
import yaml

def load_metadata(yaml_path):
    """Loads run configuration and physical constants."""
    print(f"Loading metadata from {yaml_path}...")
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_prepare(parquet_path):
    """Loads the raw parquet log and prepares the time-indexed dataframe."""
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Ensure datetime index for rolling operations
    df['timestamp'] = pd.to_datetime(df['raw.timestamp_ns'], unit='ns')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def compute_derived_metrics(df):
    """Computes the 'derived.*' namespace metrics from raw fields."""
    print("Computing derived metrics...")
    
    if 'raw.true_noise_c63' in df.columns:
        df['derived.n_est_error'] = ((df['raw.n_est_slow'] - df['raw.true_noise_c63']) / df['raw.true_noise_c63']) * 100.0
    else:
        df['derived.n_est_error'] = 0.0
        
    known_external = df['raw.known_external_heat'] if 'raw.known_external_heat' in df.columns else 0.0
    df['derived.thermal_self_load'] = df['raw.temp_center_c'] - df['raw.ambient_temp_c'] - known_external
    
    return df

def evaluate_fail_rules(df, meta):
    """Evaluates the catastrophic failure rules using metadata thresholds."""
    print("Evaluating fail parser rules...")
    failures = {}
    
    params = meta.get('test_parameters', {})
    phys = meta.get('physical_constants', {})
    
    max_runaway = params.get('max_allowable_thermal_runaway_c', 15.0)
    max_spatial = params.get('max_allowable_spatial_delta_c', 10.0)
    max_error = params.get('max_allowable_n_est_error_pct', -15.0)
    max_energy = params.get('max_allowable_energy_fj_op', 1000.0)
    tau_m_ms = phys.get('tau_m_ms', 1.0)
    
    shock_window = f"{int(100 * tau_m_ms)}ms"

    # Rule 1: Thermal Runaway
    runaway_mask = (df['raw.os_mode_state'] == 3) & (df['derived.thermal_self_load'] > max_runaway)
    failures['Thermal Runaway'] = bool(runaway_mask.any())

    # Rule 2: Silent Poison
    spatial_delta = (df['raw.temp_edge_c'] - df['raw.temp_far_c']).abs()
    poison_mask = (spatial_delta > max_spatial) & (df['derived.n_est_error'] < max_error) & (df['raw.os_mode_state'] == 9) & (df['raw.fidelity_score'] < 95.0)
    failures['Silent Poison'] = bool(poison_mask.any())

    # Rule 3: OS Thrashing
    fail_mode_thrashing = df['raw.os_mode_state'].diff().ne(0).rolling('1s').sum() > 3
    if 'raw.rollback_phase' in df.columns:
        fail_phase_thrashing = (df['raw.rollback_phase'] != df['raw.rollback_phase'].shift(1)).rolling('1s').sum() > 5
        thrashing_mask = fail_mode_thrashing | fail_phase_thrashing
    else:
        thrashing_mask = fail_mode_thrashing
    failures['OS Thrashing'] = bool(thrashing_mask.any())

    # Rule 4: False-Shock Paralysis
    # Explicit boolean cast discipline for pandas rolling operations
    if 'raw.event_marker' in df.columns and 'raw.shock_flag' in df.columns:
        has_real_shock = df['raw.event_marker'].eq('SHOCK_INJECTED').rolling(shock_window).max().fillna(0).astype(bool)
        is_paralyzed = df['raw.shock_flag'].fillna(False).astype(bool).rolling(shock_window).min().fillna(0).astype(bool)
        paralysis_mask = is_paralyzed & ~has_real_shock
        failures['False-Shock Paralysis'] = bool(paralysis_mask.any())
    else:
        failures['False-Shock Paralysis'] = False

    # Rule 5: Energy Budget Violation
    if 'raw.energy_cost_est' in df.columns:
        energy_mask = (df['raw.energy_cost_est'] > max_energy).rolling('5s').min().fillna(0).astype(bool)
        failures['Energy Budget Violation'] = bool(energy_mask.any())
    else:
        failures['Energy Budget Violation'] = False

    return failures

def extract_recovery_metrics(df):
    """Calculates recovery latency and validity around shock/disturbance events."""
    metrics = []
    
    if 'raw.event_marker' not in df.columns:
        return metrics

    dist_off_times = df[df['raw.event_marker'] == 'DISTURBANCE_OFF'].index
    
    for t_off in dist_off_times:
        future_df = df.loc[t_off:]
        rollback_complete = future_df[future_df['raw.event_marker'] == 'ROLLBACK_COMPLETE']
        
        if not rollback_complete.empty:
            t_complete = rollback_complete.index[0]
            latency_ms = (t_complete - t_off).total_seconds() * 1000.0
            fidelity_at_complete = rollback_complete['raw.fidelity_score'].iloc[0]
            is_valid = bool(fidelity_at_complete >= 99.0)
            
            metrics.append({
                'event_time': t_off,
                'latency_ms': latency_ms,
                'fidelity_at_completion': fidelity_at_complete,
                'valid': is_valid
            })
            
    return metrics

def print_report(failures, recovery_metrics):
    print("\n" + "="*50)
    print(" LEVEL 3 BENCH LOG PARSER REPORT ")
    print("="*50)
    
    has_catastrophic = any(failures.values())
    
    if has_catastrophic:
        print("\n[ STATUS: CATASTROPHIC FAILURE ]")
        for rule, failed in failures.items():
            if failed:
                print(f"  [X] {rule}")
    else:
        print("\n[ STATUS: PASS ]")
        for rule in failures.keys():
            print(f"  [ ] {rule} (Clean)")
            
    print("\n--- Recovery Latency Metrics ---")
    if not recovery_metrics:
        print("  No DISTURBANCE_OFF -> ROLLBACK_COMPLETE events found.")
    else:
        for idx, m in enumerate(recovery_metrics):
            valid_str = "VALID" if m['valid'] else "INVALID (Fidelity < 99%)"
            print(f"  Event {idx+1}: Latency = {m['latency_ms']:.1f} ms | {valid_str} | Final Fidelity = {m['fidelity_at_completion']:.1f}%")
            
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python level3_fail_parser.py <path_to_parquet_log> [path_to_metadata_yaml]")
        sys.exit(1)
        
    parquet_path = sys.argv[1]
    yaml_path = sys.argv[2] if len(sys.argv) > 2 else "level3_run_metadata.yaml"
    
    if not os.path.exists(parquet_path):
        print(f"Error: File {parquet_path} not found.")
        sys.exit(1)
        
    meta = {}
    if os.path.exists(yaml_path):
        meta = load_metadata(yaml_path)
    else:
        print(f"Warning: Metadata file {yaml_path} not found. Using default thresholds.")
        
    df = load_and_prepare(parquet_path)
    df = compute_derived_metrics(df)
    failures = evaluate_fail_rules(df, meta)
    recovery_metrics = extract_recovery_metrics(df)
    
    print_report(failures, recovery_metrics)
    
    output_path = parquet_path.replace(".parquet", "_analyzed.parquet")
    df.to_parquet(output_path)
    print(f"Saved analyzed dataset with derived metrics to: {output_path}")
