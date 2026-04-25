"""
Benchmark WZMA Ultimate - OPTIMIZED (ROCm 7.2 Tuned)
Applying learnings from Matrix_Al_challenge/paper_rocm72.md
1. Manual Attention (Already in GQA)
2. TunableOp Enabled
3. Max-Autotune Compile
"""
import os

# Set Env Vars BEFORE importing torch for ROCm tuning
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1" # From paper recommendation

import torch
import time
import sys
from torch.amp import autocast

# Add W-bit to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.wzma_reference.ultimate_model import WZMAUltimate, UltimateConfig

def bench():
    # 1. Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    config = UltimateConfig(
        vocab_size=8192,
        d_model=384,
        n_layers=4,
        n_heads=6,
        enable_plasticity=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device} (ROCm 7.2 Optimized)...")
    
    model = WZMAUltimate(config).to(device)
    
    # 2. Torch Compile with max-autotune
    print("Compiling model (max-autotune)... this may take a minute...")
    try:
        # fullgraph=False because of S4 loop and Plasticity control flow
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"Compile failed: {e}")
    
    model.train()
    
    # Input
    batch = 32
    seq_len = 128
    x = torch.randint(0, 8192, (batch, seq_len)).to(device)
    
    # Warmup (Extended for TunableOp profiling)
    print("Warmup & Tuning (100 steps)...")
    with autocast(device_type=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        for i in range(100):
            if i % 10 == 0: print(f"Warmup {i}/100")
            y = model(x)
            loss = y.mean()
            loss.backward()
            model.zero_grad()
        
    # Bench
    print("Running Measurement...")
    start = time.time()
    n_iters = 100
    
    for _ in range(n_iters):
        with autocast(device_type=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            y = model(x)
            loss = y.mean()
        
        loss.backward()
        model.zero_grad()
        
    torch.cuda.synchronize()
    end = time.time()
    
    tokens = batch * seq_len * n_iters
    duration = end - start
    tps = tokens / duration
    
    print(f"Throughput: {tps:.2f} tokens/sec")
    print(f"Latency per batch: {duration/n_iters*1000:.2f} ms")

if __name__ == "__main__":
    bench()
