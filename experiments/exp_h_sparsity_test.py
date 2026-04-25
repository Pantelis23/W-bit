import argparse
import sys
import math

def run_kv_sparsity_test():
    print("=== Experiment H: KV Access Sparsity Stress Test ===")
    print("Hypothesis: Reducing 'touched' KV (ctx_eff) restores speedup even at 128k total context.")
    
    # Constants
    d_model = 4096
    num_params = 12 * (d_model ** 2)
    bytes_w = 2 # FP16
    bytes_kv = 2 # FP16
    n_layers = 32
    
    gpu_specs = {
        "hbm_bw": 624.0 * 1e9 * 0.8,
        "compute_flopp": 50.0 * 1e12,
        "E_dram": 10.0, "E_sram": 1.0, "E_mac": 4.0
    }
    
    wbit_specs = {
        "noc_bw": 1024.0 * 1e9,
        "sram_bw": 2000.0 * 1e9,
        "hbm_bw": 624.0 * 1e9 * 0.8,
        "t_adc": 10e-9, "t_dac": 2e-9, "t_settle": 20e-9,
        "E_cell_mac": 0.05, "E_adc": 2.0, "E_dac": 0.5, "E_noc": 0.2
    }
    
    actual_tiles = max(512, math.ceil(num_params / 65536))

    def calculate_speedup(total_ctx, kv_ratio, attn_mode, window_size=None, k_retrieved=None):
        # 1. Determine Effective Context (touched per step)
        ctx_eff = total_ctx
        if attn_mode == 'window':
            ctx_eff = min(total_ctx, window_size)
        elif attn_mode == 'retrieval':
            ctx_eff = k_retrieved
            
        # 2. Workload
        # GPU fetches weights + Touched KV (assumes sparse kernel)
        # Note: Standard FlashAttention is dense. Sparse kernels exist but let's assume optimized sparse kernel.
        kv_elements_touched = 2 * 1 * ctx_eff * d_model * kv_ratio
        vol_weights = num_params * bytes_w
        vol_kv_touched = kv_elements_touched * bytes_kv
        
        # GPU Latency
        # T_mem = (Weights + KV_touched) / BW
        # T_compute based on ctx_eff
        total_macs = num_params + 2 * (ctx_eff * d_model)
        
        T_gpu_compute = total_macs / gpu_specs["compute_flopp"]
        T_gpu_mem = (vol_weights + vol_kv_touched) / gpu_specs["hbm_bw"]
        T_gpu = max(T_gpu_compute, T_gpu_mem) + 10e-6 # Sparse kernel overhead
        
        # W-bit Latency
        # Weights (Stationary)
        rounds = d_model / 256
        t_analog = wbit_specs["t_dac"] + wbit_specs["t_settle"] + wbit_specs["t_adc"]
        T_wbit_weights = (rounds * t_analog) + ((7 * d_model * 15 * 2) / wbit_specs["noc_bw"])
        
        # KV Fetch (HBM - we assume cold storage for massive 128k)
        # But we only fetch ctx_eff!
        T_wbit_kv_fetch = vol_kv_touched / wbit_specs["hbm_bw"]
        
        # Reduction overhead
        T_wbit_attn_reduce = (d_model * 4) / wbit_specs["noc_bw"] * math.log2(actual_tiles)
        
        T_wbit_attn = T_wbit_kv_fetch + T_wbit_attn_reduce + 100e-9
        T_wbit = T_wbit_weights + T_wbit_attn
        
        speedup = T_gpu / T_wbit
        
        # Sanity check
        # Speedup bound = 1 + W / KV_touched
        max_sp = 1 + (vol_weights / vol_kv_touched)
        
        return speedup, max_sp, vol_kv_touched

    print("--- Experiment H: Sparsity & Retrieval at 128k Context (MHA FP16) ---")
    print(f"{'Mode':<10} | {'Param':<8} | {'Ctx Eff':<8} | {'KV Touched':<10} | {'Sanity Max':<10} | {'Speedup':<8}")
    print("-" * 80)
    
    total_ctx = 128000
    
    # Baseline Dense
    sp, max_sp, vol = calculate_speedup(total_ctx, 1.0, 'dense')
    print(f"{'Dense':<10} | {'Full':<8} | {total_ctx:<8} | {vol/1e6:<8.1f}MB | {max_sp:<10.2f} | {sp:<8.2f}")
    
    # Windowed Attention
    windows = [32768, 16384, 8192]
    for w in windows:
        sp, max_sp, vol = calculate_speedup(total_ctx, 1.0, 'window', window_size=w)
        print(f"{'Window':<10} | {w:<8} | {w:<8} | {vol/1e6:<8.1f}MB | {max_sp:<10.2f} | {sp:<8.2f}")
        
    # Retrieval Augmented Generation (RAG) within Context
    retrieved_k = [4096, 2048, 512]
    for k in retrieved_k:
        sp, max_sp, vol = calculate_speedup(total_ctx, 1.0, 'retrieval', k_retrieved=k)
        print(f"{'Retrieval':<10} | {k:<8} | {k:<8} | {vol/1e6:<8.1f}MB | {max_sp:<10.2f} | {sp:<8.2f}")

if __name__ == "__main__":
    run_kv_sparsity_test()
