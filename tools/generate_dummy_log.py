import pandas as pd
import numpy as np
import os

def generate_dummy_log(output_path="dummy_bench_run.parquet"):
    """Generates a dummy Level 3 bench log to test the parser and plotting infrastructure."""
    
    print("Generating dummy hardware log...")
    
    # Simulate 2 minutes of data at 10 Hz (100ms intervals) -> 1200 rows
    num_rows = 1200
    
    # 1. Metadata
    # Start at a realistic epoch
    start_time = pd.Timestamp("2026-03-16 10:00:00").value
    timestamps = np.linspace(start_time, start_time + (120 * 1e9), num_rows, dtype=np.int64)
    
    df = pd.DataFrame({
        'raw.timestamp_ns': timestamps,
        'raw.tile_id': ['T_0_0'] * num_rows,
        'raw.event_marker': [None] * num_rows
    })
    
    # 2. Control State & Events
    # 0-30s: R=9 (Steady)
    # 30s: DISTURBANCE_ON
    # 30-40s: R=3 (Downgrade -> Hold)
    # 40s: DISTURBANCE_OFF
    # 40-50s: R=3 (Hold)
    # 50-60s: R=5 (step_3_5)
    # 60-70s: R=7 (step_5_7)
    # 70-80s: R=9 (step_7_9 -> recenter)
    # 80-120s: R=9 (Steady)
    
    modes = np.full(num_rows, 9, dtype=np.int8)
    phases = np.full(num_rows, 'steady', dtype=object)
    
    # Event timestamps (approx row indices)
    idx_dist_on = 300
    idx_dist_off = 400
    idx_roll_start = 500
    idx_step_5 = 600
    idx_step_7 = 700
    idx_recenter = 800
    idx_roll_complete = 820
    
    df.loc[idx_dist_on, 'raw.event_marker'] = 'DISTURBANCE_ON'
    df.loc[idx_dist_off, 'raw.event_marker'] = 'DISTURBANCE_OFF'
    df.loc[idx_roll_start, 'raw.event_marker'] = 'ROLLBACK_START'
    df.loc[idx_recenter, 'raw.event_marker'] = 'RECENTER_WRITE_START'
    df.loc[idx_roll_complete, 'raw.event_marker'] = 'ROLLBACK_COMPLETE'
    
    # Apply modes
    modes[idx_dist_on:idx_roll_start] = 3
    modes[idx_roll_start:idx_step_5] = 5
    modes[idx_step_5:idx_step_7] = 7
    
    # Apply phases
    phases[idx_dist_on:idx_roll_start] = 'hold'
    phases[idx_dist_on:idx_dist_on+5] = 'downgrade'
    phases[idx_roll_start:idx_step_5] = 'step_3_5'
    phases[idx_step_5:idx_step_7] = 'step_5_7'
    phases[idx_step_7:idx_recenter] = 'step_7_9'
    phases[idx_recenter:idx_roll_complete] = 'recenter'
    
    df['raw.os_mode_state'] = modes
    df['raw.rollback_phase'] = phases
    
    # 3. Tradeoff Vector
    fidelity = np.full(num_rows, 100.0, dtype=np.float32)
    # Slight dip during disturbance before downgrade completes
    fidelity[idx_dist_on:idx_dist_on+2] = 96.0
    df['raw.fidelity_score'] = fidelity
    
    energy = np.full(num_rows, 40.0, dtype=np.float32)
    energy[modes == 3] = 516.0
    energy[modes == 5] = 300.0
    energy[modes == 7] = 150.0
    # Add random noise to energy
    energy += np.random.normal(0, 2.0, num_rows)
    df['raw.energy_cost_est'] = energy
    
    df['raw.boundary_bin_err'] = np.where(modes < 9, np.random.normal(0.5, 0.1, num_rows), 0.0)
    df['raw.boundary_bin_err'] = np.clip(df['raw.boundary_bin_err'], 0, 100)
    
    # 4. Environment Variables
    df['raw.ambient_temp_c'] = np.full(num_rows, 25.0, dtype=np.float32)
    df['raw.known_external_heat'] = np.where((np.arange(num_rows) >= idx_dist_on) & (np.arange(num_rows) <= idx_dist_off), 20.0, 0.0)
    
    # Temperature dynamics (simulating thermal mass)
    temp_center = np.full(num_rows, 25.0)
    temp_edge = np.full(num_rows, 25.0)
    temp_far = np.full(num_rows, 25.0)
    
    for i in range(1, num_rows):
        # Base self-heating based on energy cost
        self_heat = (energy[i] - 40.0) / 100.0
        target_center = 25.0 + self_heat + df['raw.known_external_heat'].iloc[i]
        
        # Leaky integrator for temperature
        temp_center[i] = temp_center[i-1] + (target_center - temp_center[i-1]) * 0.05
        
        # Edge and far follow center with some gradient
        temp_edge[i] = temp_center[i] + np.random.normal(0, 0.5)
        temp_far[i] = temp_center[i] - 1.0 + np.random.normal(0, 0.5)
        
    df['raw.temp_center_c'] = temp_center.astype(np.float32)
    df['raw.temp_edge_c'] = temp_edge.astype(np.float32)
    df['raw.temp_far_c'] = temp_far.astype(np.float32)
    
    # Estimator
    true_noise = np.full(num_rows, 1.0, dtype=np.float32)
    true_noise[idx_dist_on:idx_dist_off] = 5.0
    # Simulate estimator lag
    n_est = np.copy(true_noise)
    for i in range(1, num_rows):
        n_est[i] = n_est[i-1] + (true_noise[i] - n_est[i-1]) * 0.1
    
    df['raw.true_noise_c63'] = true_noise
    df['raw.n_est_slow'] = n_est
    
    # 5. Parasitics & Shocks
    df['raw.shock_flag'] = np.zeros(num_rows, dtype=bool)
    # Add a brief shock
    df.loc[100, 'raw.event_marker'] = 'SHOCK_INJECTED'
    df.loc[100:105, 'raw.shock_flag'] = True
    df.loc[106, 'raw.event_marker'] = 'SHOCK_CLEAR'
    
    df['raw.v_offset_drift'] = np.cumsum(np.random.normal(0, 0.1, num_rows))
    df['raw.beta_far_ratio'] = np.where(modes < 9, np.random.normal(0.85, 0.02, num_rows), 1.0)
    df['raw.adj_leak_ratio'] = np.where(modes < 9, np.random.normal(0.02, 0.01, num_rows), 0.0)

    # Save
    df.to_parquet(output_path)
    print(f"Dummy log successfully saved to {output_path}")

if __name__ == "__main__":
    generate_dummy_log()
