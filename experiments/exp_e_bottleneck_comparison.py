import argparse
import sys
import math

def rigorous_bottleneck_comparison(args):
    print("=== Experiment E (Rigorous v3): Full System Bottleneck Comparison ===")
    print(f"Workload: Transformer Decode Step (B=1, L=1, Context={args.context})")
    print(f"Precision: {args.prec}, KV Location: {args.kv_loc}")
    
    # --- 1. Workload Definition ---
    d_model = 4096
    context_len = args.context
    batch_size = 1
    
    if args.prec == 'fp32':
        bytes_w, bytes_kv, bytes_accum = 4, 4, 4
    elif args.prec == 'fp16':
        bytes_w, bytes_kv, bytes_accum = 2, 2, 2
    elif args.prec == 'int8':
        bytes_w, bytes_kv, bytes_accum = 1, 1, 4 # Accumulate in higher prec
    else:
        bytes_w, bytes_kv, bytes_accum = 2, 2, 2
        
    num_params = 12 * (d_model ** 2)
    kv_elements = 2 * batch_size * context_len * d_model
    
    compute_macs_weights = num_params 
    compute_macs_attn = 2 * (context_len * d_model)
    total_macs = compute_macs_weights + compute_macs_attn
    
    print(f"Model: d={d_model}, Context={context_len}")
    print(f"  Params: {num_params/1e6:.2f} M")
    print(f"  KV Cache: {kv_elements*bytes_kv/1e6:.2f} MB")
    print("-" * 50)

    # --- 2. Architecture Definitions ---

    gpu_specs = {
        "name": "Consumer GPU (Optimized)",
        "hbm_bw": 624.0 * 1e9 * 0.8,
        "compute_flopp": 50.0 * 1e12,
        "E_dram": 10.0, "E_sram": 1.0, "E_mac": 4.0,
    }

    wbit_specs = {
        "name": "W-bit Fabric (Physically Grounded)",
        "tile_size": (256, 256),
        "cells_per_tile": 256*256,
        
        # Analog Timings
        "t_adc": 10e-9, "t_dac": 2e-9, "t_settle": 20e-9,
        
        # Bandwidths
        "noc_bw": 1024.0 * 1e9,     # 1 TB/s NoC
        "sram_bw": 2000.0 * 1e9,    # Local SRAM
        "hbm_bw": 624.0 * 1e9 * 0.8, # Fallback to HBM if KV fits there
        
        # Energies
        "E_cell_mac": 0.05, "E_adc": 2.0, "E_dac": 0.5, "E_noc": 0.2, "E_sram": 1.0, "E_dram": 10.0
    }

    # --- 3. GPU Simulation ---
    print(f"Running Simulation: {gpu_specs['name']}...")
    T_gpu_compute = total_macs / gpu_specs["compute_flopp"]
    vol_weights = num_params * bytes_w
    vol_kv = kv_elements * bytes_kv
    
    T_gpu_weight_fetch = vol_weights / gpu_specs["hbm_bw"]
    T_gpu_kv_fetch = vol_kv / gpu_specs["hbm_bw"]
    
    T_gpu_total = max(T_gpu_compute, T_gpu_weight_fetch + T_gpu_kv_fetch) + 5e-6
    E_gpu_total = (vol_weights + vol_kv) * 8 * gpu_specs["E_dram"] + total_macs * gpu_specs["E_mac"] + (total_macs * 4 * 16) * gpu_specs["E_sram"]
    P_gpu = E_gpu_total / T_gpu_total * 1e-12

    print(f"  Latency: {T_gpu_total*1e6:.2f} us")
    print(f"  Power:   {P_gpu:.2f} W")
    print(f"  Breakdown: Weight_Fetch={T_gpu_weight_fetch*1e6:.2f}us, KV={T_gpu_kv_fetch*1e6:.2f}us")
    
    # --- 4. W-bit Simulation ---
    print(f"\nRunning Simulation: {wbit_specs['name']}...")
    
    needed_cells = num_params
    needed_tiles = math.ceil(needed_cells / wbit_specs["cells_per_tile"])
    actual_tiles = max(512, needed_tiles)
    
    # A. Weights Compute
    # Problem 2: Tile Ops & ADC Cycles
    # We have d_model inputs. Tile width is 256.
    # Broadcast rounds = d_model / 256 = 16 rounds.
    # In each round, all tiles fire (if mapped correctly).
    # Time = Rounds * (t_dac + t_settle + t_adc)
    rounds_per_gemv = d_model / 256
    t_analog_cycle = wbit_specs["t_dac"] + wbit_specs["t_settle"] + wbit_specs["t_adc"]
    T_wbit_compute_core = rounds_per_gemv * t_analog_cycle
    
    # Problem 1: Reduction Traffic
    # Output d_model elements.
    # Matrix size d * d. Split into 16 * 16 grid of tiles.
    # Each output element accumulates partial sums from 16 tiles (k_splits).
    k_splits = d_model // 256
    # For EACH GEMV (Q,K,V,O, Gate, Up, Down -> 7 matrices)
    # Traffic = 7 * d_model * (k_splits - 1) * bytes_accum
    num_gemvs = 7
    vol_reduce = num_gemvs * d_model * (k_splits - 1) * bytes_accum
    T_wbit_reduce = vol_reduce / wbit_specs["noc_bw"]
    
    T_wbit_weights = T_wbit_compute_core + T_wbit_reduce
    
    # B. Attention Compute
    # Problem 3: KV Location
    if args.kv_loc == 'hbm':
        # Fetch KV from off-chip HBM
        T_wbit_kv_fetch = vol_kv / wbit_specs["hbm_bw"]
        E_wbit_kv_fetch = vol_kv * 8 * wbit_specs["E_dram"]
    else:
        # Fetch from local SRAM
        T_wbit_kv_fetch = vol_kv / wbit_specs["sram_bw"]
        E_wbit_kv_fetch = vol_kv * 8 * wbit_specs["E_sram"]
        
    T_wbit_softmax_reduce = 100e-9
    T_wbit_attn_reduce = (d_model * 4) / wbit_specs["noc_bw"] * math.log2(actual_tiles)
    
    T_wbit_attn = T_wbit_kv_fetch + T_wbit_softmax_reduce + T_wbit_attn_reduce
    
    T_wbit_total = T_wbit_weights + T_wbit_attn
    
    # Energy
    E_wbit_dac = d_model * num_gemvs * wbit_specs["E_dac"]
    E_wbit_mac = compute_macs_weights * wbit_specs["E_cell_mac"]
    E_wbit_adc = d_model * num_gemvs * k_splits * wbit_specs["E_adc"] # Every partial sum is digitized
    E_wbit_noc = vol_reduce * 8 * wbit_specs["E_noc"]
    E_wbit_attn_compute = compute_macs_attn * 1.0
    
    E_wbit_total = E_wbit_dac + E_wbit_mac + E_wbit_adc + E_wbit_noc + E_wbit_kv_fetch + E_wbit_attn_compute
    P_wbit = E_wbit_total / T_wbit_total * 1e-12
    
    print(f"  Latency: {T_wbit_total*1e6:.2f} us")
    print(f"  Power:   {P_wbit:.2f} W")
    print(f"  Breakdown:")
    print(f"    Weights Compute: {T_wbit_compute_core*1e6:.2f} us")
    print(f"    Weights Reduce:  {T_wbit_reduce*1e6:.2f} us (NoC traffic)")
    print(f"    KV Fetch ({args.kv_loc.upper()}): {T_wbit_kv_fetch*1e6:.2f} us")
    
    # --- 5. Comparison ---
    print("-" * 50)
    lat_speedup = T_gpu_total / T_wbit_total
    energy_saving = E_gpu_total / E_wbit_total
    
    print(f"RESULTS ({args.prec.upper()}, KV={args.kv_loc.upper()}):")
    print(f"Latency Speedup: {lat_speedup:.2f}x")
    print(f"Energy Saving:   {energy_saving:.2f}x")
    
    # Capacity Sanity
    kv_mb = kv_elements * bytes_kv / 1e6
    if args.kv_loc == 'local_sram':
        print(f"\nIMPLICATION: You need {kv_mb:.1f} MB of SRAM per layer.")
        print(f"Total for 32 layers: {kv_mb*32/1000:.2f} GB on-chip SRAM. (Expensive!)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prec', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'])
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--kv_loc', type=str, default='local_sram', choices=['local_sram', 'hbm'])
    args = parser.parse_args()
    rigorous_bottleneck_comparison(args)