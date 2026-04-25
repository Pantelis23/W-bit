import argparse
import sys
import math

def run_stress_test():
    print("=== Experiment F: W-bit Stress Tests (Context & Batch) ===")
    
    # Constants
    d_model = 4096
    num_params = 12 * (d_model ** 2)
    bytes_w = 2 # FP16
    bytes_kv = 2 # FP16
    bytes_accum = 2
    
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
        "hbm_bw": 624.0 * 1e9 * 0.8, # Shared HBM for overflow
        "t_adc": 10e-9, "t_dac": 2e-9, "t_settle": 20e-9,
        "E_cell_mac": 0.05, "E_adc": 2.0, "E_dac": 0.5, "E_noc": 0.2, "E_sram": 1.0, "E_dram": 10.0
    }
    
    actual_tiles = max(512, math.ceil(num_params / 65536))

    # --- SWEEP 1: Context Length (B=1) ---
    print("\n--- SWEEP 1: Context Length (Batch=1) ---")
    contexts = [2048, 8192, 32768, 128000]
    
    print(f"{'Context':<10} | {'GPU Lat (us)':<12} | {'W-bit Lat (us)':<14} | {'Speedup':<8} | {'KV Size (MB)':<12}")
    print("-" * 70)
    
    for ctx in contexts:
        # GPU
        kv_elements = 2 * 1 * ctx * d_model
        total_macs = num_params + 2 * (ctx * d_model)
        
        T_gpu_compute = total_macs / gpu_specs["compute_flopp"]
        T_gpu_mem = (num_params * bytes_w + kv_elements * bytes_kv) / gpu_specs["hbm_bw"]
        T_gpu = max(T_gpu_compute, T_gpu_mem) + 5e-6
        
        # W-bit
        # Weights
        rounds = d_model / 256
        t_analog = wbit_specs["t_dac"] + wbit_specs["t_settle"] + wbit_specs["t_adc"]
        T_wbit_weights_compute = rounds * t_analog
        
        k_splits = d_model // 256
        vol_reduce = 7 * d_model * (k_splits - 1) * bytes_accum
        T_wbit_weights_reduce = vol_reduce / wbit_specs["noc_bw"]
        T_wbit_weights = T_wbit_weights_compute + T_wbit_weights_reduce
        
        # KV (Assume HBM overflow for > 1GB)
        kv_bytes = kv_elements * bytes_kv
        if kv_bytes > 1e9: # >1GB, use HBM
            T_wbit_kv = kv_bytes / wbit_specs["hbm_bw"]
        else: # Use SRAM
            T_wbit_kv = kv_bytes / wbit_specs["sram_bw"]
            
        T_wbit_attn_reduce = (d_model * 4) / wbit_specs["noc_bw"] * math.log2(actual_tiles)
        T_wbit_attn = T_wbit_kv + T_wbit_attn_reduce + 100e-9
        
        T_wbit = T_wbit_weights + T_wbit_attn
        
        speedup = T_gpu / T_wbit
        print(f"{ctx:<10} | {T_gpu*1e6:<12.1f} | {T_wbit*1e6:<14.1f} | {speedup:<8.1f} | {kv_bytes/1e6:<12.1f}")

    # --- SWEEP 2: Batch Size (Context=2048) ---
    print("\n--- SWEEP 2: Batch Size (Context=2048) ---")
    batches = [1, 4, 16, 64, 128]
    
    print(f"{'Batch':<10} | {'GPU Lat (us)':<12} | {'W-bit Lat (us)':<14} | {'Speedup':<8} | {'GPU T-put':<10}")
    print("-" * 75)
    
    ctx = 2048
    for B in batches:
        # GPU
        # Weights fetched ONCE per batch step
        kv_elements = 2 * B * ctx * d_model
        total_macs = B * num_params + B * 2 * (ctx * d_model)
        
        T_gpu_compute = total_macs / gpu_specs["compute_flopp"]
        # Weight fetch amortized over B? No, fetched once per step.
        # Data = Weights + B * KV
        T_gpu_mem = (num_params * bytes_w + kv_elements * bytes_kv) / gpu_specs["hbm_bw"]
        T_gpu = max(T_gpu_compute, T_gpu_mem) + 5e-6
        
        # W-bit
        # Weights: Process B vectors?
        # Option A: Sequential (B times) -> Latency linear with B.
        # Option B: Parallel (if tiles support multiple row activations?) No.
        # Option C: Pipelined? 
        # Standard CIM handles B=1 best. For B>1, we iterate or pipeline.
        # Let's assume sequential processing of B vectors for weights.
        # T_weights = B * (Compute + Reduce)
        T_wbit_weights = B * (T_wbit_weights_compute + T_wbit_weights_reduce)
        
        # KV Fetch
        kv_bytes = kv_elements * bytes_kv
        # If B is large, KV might force HBM
        if kv_bytes > 1e9:
            T_wbit_kv = kv_bytes / wbit_specs["hbm_bw"]
        else:
            T_wbit_kv = kv_bytes / wbit_specs["sram_bw"]
            
        T_wbit_attn = T_wbit_kv + (B * T_wbit_attn_reduce) # Reduce per query
        
        T_wbit = T_wbit_weights + T_wbit_attn
        
        speedup = T_gpu / T_wbit
        gpu_tput = B / T_gpu # tokens/sec (layers/sec)
        
        print(f"{B:<10} | {T_gpu*1e6:<12.1f} | {T_wbit*1e6:<14.1f} | {speedup:<8.1f} | {gpu_tput:<10.1f}")

if __name__ == "__main__":
    run_stress_test()