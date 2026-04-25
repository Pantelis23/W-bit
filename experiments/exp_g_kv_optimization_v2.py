import argparse
import sys
import math

def run_kv_tests_v2(args):
    print("=== Experiment G (Rigorous v2): No-Cheating KV Stress Tests ===")
    
    # Base Constants
    d_model = 4096
    n_layers = 32 # Required to calculate total SRAM pressure
    num_params_per_layer = 12 * (d_model ** 2)
    bytes_w = 2 # FP16 Weights (Stationary)
    
    # Architectures
    gpu_specs = {
        "name": "GPU",
        "hbm_bw": 624.0 * 1e9 * 0.8,
        "compute_flopp": 50.0 * 1e12,
        "E_dram": 10.0, "E_sram": 1.0, "E_mac": 4.0
    }
    
    wbit_specs = {
        "name": "W-bit",
        "tile_size": (256, 256),
        "cells_per_tile": 65536,
        "noc_bw": 1024.0 * 1e9,
        "sram_bw": 2000.0 * 1e9,
        "hbm_bw": 624.0 * 1e9 * 0.8, 
        "t_adc": 10e-9, "t_dac": 2e-9, "t_settle": 20e-9,
    }
    
    actual_tiles = max(512, math.ceil(num_params_per_layer / 65536))

    def calculate_speedup(ctx, kv_ratio, bytes_kv, kv_loc_override=None):
        # 1. Volumetrics
        kv_elements_layer = 2 * 1 * ctx * d_model * kv_ratio
        vol_weights_layer = num_params_per_layer * bytes_w
        vol_kv_layer = kv_elements_layer * bytes_kv
        
        vol_kv_total = vol_kv_layer * n_layers
        
        # 2. Hard Sanity Bound (KV in HBM limit)
        # Speedup Max = (W + KV) / KV = 1 + W/KV
        max_speedup_hbm = 1.0 + (vol_weights_layer / vol_kv_layer)
        
        # 3. GPU Latency
        total_macs = num_params_per_layer + 2 * (ctx * d_model)
        T_gpu_compute = total_macs / gpu_specs["compute_flopp"]
        T_gpu_mem = (vol_weights_layer + vol_kv_layer) / gpu_specs["hbm_bw"]
        T_gpu = max(T_gpu_compute, T_gpu_mem) + 5e-6
        
        # 4. W-bit Latency
        # Weights (Stationary)
        rounds = d_model / 256
        t_analog = wbit_specs["t_dac"] + wbit_specs["t_settle"] + wbit_specs["t_adc"]
        T_wbit_weights_compute = rounds * t_analog
        
        k_splits = d_model // 256
        vol_reduce = 7 * d_model * (k_splits - 1) * 2
        T_wbit_weights_reduce = vol_reduce / wbit_specs["noc_bw"]
        T_wbit_weights = T_wbit_weights_compute + T_wbit_weights_reduce
        
        # KV Fetch Placement
        loc = kv_loc_override or args.kv_loc
        
        sram_budget_bytes = args.sram_budget_mb * 1e6
        
        vol_sram_fetch = 0.0
        vol_hbm_fetch = 0.0
        
        if loc == 'sram':
            if vol_kv_total > sram_budget_bytes:
                # Fallback to HBM if budget exceeded
                vol_hbm_fetch = vol_kv_layer
            else:
                vol_sram_fetch = vol_kv_layer
        elif loc == 'hbm':
            vol_hbm_fetch = vol_kv_layer
        elif loc == 'hier':
            # Fill SRAM first
            if vol_kv_total <= sram_budget_bytes:
                vol_sram_fetch = vol_kv_layer
            else:
                # Fraction that fits
                fraction_sram = sram_budget_bytes / vol_kv_total
                vol_sram_fetch = vol_kv_layer * fraction_sram
                vol_hbm_fetch = vol_kv_layer * (1.0 - fraction_sram)
        
        T_kv_fetch = (vol_sram_fetch / wbit_specs["sram_bw"]) + (vol_hbm_fetch / wbit_specs["hbm_bw"])
        
        T_wbit_attn_reduce = (d_model * 4) / wbit_specs["noc_bw"] * math.log2(actual_tiles)
        T_wbit_attn = T_kv_fetch + T_wbit_attn_reduce + 100e-9
        
        T_wbit = T_wbit_weights + T_wbit_attn
        
        speedup = T_gpu / T_wbit
        
        return speedup, max_speedup_hbm, vol_kv_total

    # --- TEST 1: GQA/MQA (FP16) ---
    print("\n--- TEST 1: GQA/MQA Effects (FP16) ---")
    print(f"{'Context':<8} | {'Arch':<8} | {'KV Total':<10} | {'Sanity Max':<10} | {'Speedup':<8} | {'Limit?':<8}")
    print("-" * 75)
    
    contexts = [32768, 128000]
    ratios = [1.0, 0.125, 0.03125] 
    names = {1.0: "MHA", 0.125: "GQA", 0.03125: "MQA"}
    
    for ctx in contexts:
        for r in ratios:
            sp, max_sp, vol = calculate_speedup(ctx, r, 2.0)
            limit_hit = "HBM" if sp > max_sp * 0.95 and args.kv_loc == 'hbm' else "NoC"
            print(f"{ctx:<8} | {names[r]:<8} | {vol/1e9:<8.2f}GB | {max_sp:<10.2f} | {sp:<8.2f} | {limit_hit:<8}")

    # --- TEST 2: KV Quantization (128k, GQA) ---
    print("\n--- TEST 2: KV Quantization (128k, GQA) ---")
    print(f"{'Prec':<8} | {'KV Total':<10} | {'Sanity Max':<10} | {'Speedup':<8}")
    print("-" * 60)
    
    ctx = 128000
    r = 0.125 # GQA
    precs = [2.0, 1.0, 0.5]
    pnames = {2.0: "FP16", 1.0: "INT8", 0.5: "INT4"}
    
    for p in precs:
        sp, max_sp, vol = calculate_speedup(ctx, r, p)
        print(f"{pnames[p]:<8} | {vol/1e9:<8.2f}GB | {max_sp:<10.2f} | {sp:<8.2f}")

    # --- TEST 3: Hot Window (128k, MHA) ---
    print("\n--- TEST 3: Hierarchy Effectiveness (128k, MHA) ---")
    print(f"SRAM Budget: {args.sram_budget_mb} MB")
    # This simulation assumes 'hier' tries to fit as much as possible.
    # Hot Window is implied by the SRAM budget.
    
    sp, max_sp, vol = calculate_speedup(128000, 1.0, 2.0, kv_loc_override='hier')
    fraction_sram = min(1.0, (args.sram_budget_mb * 1e6) / vol)
    print(f"KV Total: {vol/1e9:.2f} GB")
    print(f"Hot Fraction: {fraction_sram*100:.1f}%")
    print(f"Speedup: {sp:.2f}x (Max HBM: {max_sp:.2f}x)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kv_loc', type=str, default='hbm', choices=['hbm', 'sram', 'hier'])
    parser.add_argument('--sram_budget_mb', type=float, default=512.0)
    args = parser.parse_args()
    run_kv_tests_v2(args)
