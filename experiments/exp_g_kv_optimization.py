import argparse
import sys
import math

def run_kv_tests():
    print("=== Experiment G: KV-Wall Mitigation Stress Tests ===")
    
    # Base Constants
    d_model = 4096
    n_heads = 32
    num_params = 12 * (d_model ** 2)
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
    
    actual_tiles = max(512, math.ceil(num_params / 65536))

    def calculate_speedup(ctx, kv_ratio, bytes_kv, hot_window=None):
        kv_elements = 2 * 1 * ctx * d_model * kv_ratio
        vol_weights = num_params * bytes_w
        vol_kv = kv_elements * bytes_kv
        
        total_macs = num_params + 2 * (ctx * d_model)
        
        T_gpu_compute = total_macs / gpu_specs["compute_flopp"]
        T_gpu_mem = (vol_weights + vol_kv) / gpu_specs["hbm_bw"]
        T_gpu = max(T_gpu_compute, T_gpu_mem) + 5e-6
        
        rounds = d_model / 256
        t_analog = wbit_specs["t_dac"] + wbit_specs["t_settle"] + wbit_specs["t_adc"]
        T_wbit_weights_compute = rounds * t_analog
        
        k_splits = d_model // 256
        vol_reduce = 7 * d_model * (k_splits - 1) * 2
        T_wbit_weights_reduce = vol_reduce / wbit_specs["noc_bw"]
        T_wbit_weights = T_wbit_weights_compute + T_wbit_weights_reduce
        
        if hot_window is not None:
            if hot_window >= ctx:
                T_kv_fetch = vol_kv / wbit_specs["sram_bw"]
            else:
                hot_ratio = hot_window / ctx
                vol_hot = vol_kv * hot_ratio
                vol_cold = vol_kv * (1.0 - hot_ratio)
                T_kv_fetch = (vol_hot / wbit_specs["sram_bw"]) + (vol_cold / wbit_specs["hbm_bw"])
        else:
            if vol_kv > 1e9:
                T_kv_fetch = vol_kv / wbit_specs["hbm_bw"]
            else:
                T_kv_fetch = vol_kv / wbit_specs["sram_bw"]
                
        T_wbit_attn_reduce = (d_model * 4) / wbit_specs["noc_bw"] * math.log2(actual_tiles)
        T_wbit_attn = T_kv_fetch + T_wbit_attn_reduce + 100e-9
        
        T_wbit = T_wbit_weights + T_wbit_attn
        
        return T_gpu, T_wbit

    # --- TEST 1: GQA/MQA (Reducing KV Heads) ---
    print("\n--- TEST 1: GQA/MQA Effects (FP16) ---")
    print(f"{'Context':<8} | {'Arch':<8} | {'KV Ratio':<8} | {'GPU (us)':<10} | {'W-bit (us)':<10} | {'Speedup':<8} | {'KV Size':<10}")
    print("-" * 85)
    
    contexts = [32768, 128000]
    ratios = [1.0, 0.125, 0.03125]
    names = {1.0: "MHA", 0.125: "GQA", 0.03125: "MQA"}
    
    for ctx in contexts:
        for r in ratios:
            t_g, t_w = calculate_speedup(ctx, r, 2.0)
            kv_mb = (2 * ctx * d_model * r * 2) / 1e6
            print(f"{ctx:<8} | {names[r]:<8} | {r:<8.3f} | {t_g*1e6:<10.1f} | {t_w*1e6:<10.1f} | {t_g/t_w:<8.1f} | {kv_mb:<8.1f}MB")

    # --- TEST 2: KV Quantization ---
    print("\n--- TEST 2: KV Quantization (MHA, 32k) ---")
    print(f"{'Context':<8} | {'Prec':<8} | {'Bytes':<8} | {'GPU (us)':<10} | {'W-bit (us)':<10} | {'Speedup':<8} | {'KV Size':<10}")
    print("-" * 85)
    
    ctx = 32768
    precs = [2.0, 1.0, 0.5]
    pnames = {2.0: "FP16", 1.0: "INT8", 0.5: "INT4"}
    
    for p in precs:
        t_g, t_w = calculate_speedup(ctx, 1.0, p)
        kv_mb = (2 * ctx * d_model * 1.0 * p) / 1e6
        print(f"{ctx:<8} | {pnames[p]:<8} | {p:<8.1f} | {t_g*1e6:<10.1f} | {t_w*1e6:<10.1f} | {t_g/t_w:<8.1f} | {kv_mb:<8.1f}MB")

    # --- TEST 3: Hot Window Hierarchy (128k Context, MHA, FP16) ---
    print("\n--- TEST 3: Hot Window Hierarchy (128k, MHA, FP16) ---")
    print("Baseline 128k Speedup was ~1.2x. Can local SRAM save it?")
    print(f"{'Hot Win':<10} | {'SRAM Req':<10} | {'W-bit (us)':<10} | {'Speedup':<8}")
    print("-" * 60)
    
    ctx = 128000
    hot_windows = [0, 1024, 4096, 16384]
    
    for hot in hot_windows:
        t_g, t_w = calculate_speedup(ctx, 1.0, 2.0, hot_window=hot)
        sram_req = (2 * hot * d_model * 2) / 1e6
        print(f"{hot:<10} | {sram_req:<8.1f}MB | {t_w*1e6:<10.1f} | {t_g/t_w:<8.1f}")

    # --- COMBINED: The "Production" Configuration ---
    print("\n--- TEST 4: The 'Production' Config (128k, GQA, INT8, 4k Hot) ---")
    ctx = 128000
    t_g, t_w = calculate_speedup(ctx, 0.125, 1.0, hot_window=4096)
    print(f"Context: {ctx}")
    print(f"Config: GQA (1/8) + INT8 + 4k SRAM Window")
    print(f"GPU Latency: {t_g*1e6:.1f} us")
    print(f"W-bit Latency: {t_w*1e6:.1f} us")
    print(f"Final Speedup: {t_g/t_w:.2f}x")

if __name__ == "__main__":
    run_kv_tests()