import argparse
import sys
import os
import random
import math
import statistics

# Add parent directory to path to import wbit
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wbit.analog_network import AnalogWbitNetwork

def generate_lora_patch(dim_in, dim_out, rank, seed):
    rng = random.Random(seed)
    # A: [dim_in, rank], B: [rank, dim_out]
    A = [[rng.gauss(0, 1.0) for _ in range(rank)] for _ in range(dim_in)]
    B = [[rng.gauss(0, 1.0) for _ in range(dim_out)] for _ in range(rank)]
    return A, B

def flatten_weights(A, B):
    # Flatten A and B into a single vector
    flat = []
    for row in A: flat.extend(row)
    for row in B: flat.extend(row)
    return flat

def quantize_to_wbit(value, n):
    # Quantize float value to range [-n, +n]
    # Assume value is roughly N(0, 1), clip to [-3, 3] then map
    clipped = max(-3.0, min(3.0, value))
    # Map [-3, 3] -> [-n, n]
    scaled = (clipped / 3.0) * n
    # Round to nearest integer level
    level = round(scaled)
    return level

def dequantize_from_wbit(level, n):
    # Map [-n, n] back to [-3, 3]
    return (level / n) * 3.0

def run_wzma_experiment(args):
    print(f"=== Experiment D: WZMA Patch Reconstruction on W-bit ===")
    print(f"Configuration: Mode={args.mode}, R={args.R} (n={(args.R-1)//2}), Noise Sigma={args.sigma}")
    
    dim_in = 16
    dim_out = 16
    rank = 4
    total_params = (dim_in * rank) + (rank * dim_out) # 64 + 64 = 128 params
    
    n_levels = (args.R - 1) // 2
    if n_levels < 1: n_levels = 1 # Safety
    
    similarities = []
    
    for trial in range(args.trials):
        trial_seed = args.seed + trial
        
        # 1. Generate "True" Patch (The Knowledge)
        A_true, B_true = generate_lora_patch(dim_in, dim_out, rank, trial_seed)
        flat_true = flatten_weights(A_true, B_true)
        
        # 2. Encode/Store in W-bit Substrate
        # Each W-bit cell stores one quantized weight parameter
        # In a real WZMA, we'd use VQ, but for this substrate test, scalar quantization is a good proxy
        stored_state = []
        for val in flat_true:
            q = quantize_to_wbit(val, n_levels)
            stored_state.append(q)
            
        # 3. Simulate Storage Noise (The "Analog" part)
        # W-bit cells drift or have read noise
        # We simulate this by adding Gaussian noise to the discrete level index
        # and then re-quantizing (or reading as float if soft-read)
        
        noisy_readout = []
        rng_noise = random.Random(trial_seed + 1000)
        
        for q_val in stored_state:
            # Noise is added to the level index itself (e.g. level 2 becomes 2.1 or 1.9)
            # In W-bit physics, state is continuous until read.
            analog_val = float(q_val) + rng_noise.gauss(0, args.sigma)
            
            if args.mode == 'binary':
                # Binary mode forces hard snap to nearest valid level (0 or 1 usually, but here mapped to range)
                # If R=2, levels are roughly mapped. 
                # To be fair to "Binary Baseline", we assume R=2 means 1-bit quantization.
                # n_levels for binary effectively acts as 1 (sign bit).
                # But here we stick to the args.R passed.
                # Hard snap to integer level
                read_val = round(analog_val)
                # Clamp to valid range [-n, n]
                read_val = max(-n_levels, min(n_levels, read_val))
            else:
                # W-bit mode allows "soft" reads? 
                # If Aeternum reads digital, it clamps to int.
                # If we use "Analog Decompression", we might use the float value directly!
                # Let's assume Aeternum digital read for now (clamped int).
                read_val = round(analog_val)
                read_val = max(-n_levels, min(n_levels, read_val))
            
            noisy_readout.append(read_val)
            
        # 4. Decode
        flat_recon = [dequantize_from_wbit(x, n_levels) for x in noisy_readout]
        
        # 5. Measure Fidelity (Cosine Similarity of the weight vector)
        # Dot product
        dot = sum(t * r for t, r in zip(flat_true, flat_recon))
        norm_t = math.sqrt(sum(t*t for t in flat_true))
        norm_r = math.sqrt(sum(r*r for r in flat_recon))
        
        if norm_r == 0:
            sim = 0.0
        else:
            sim = dot / (norm_t * norm_r)
            
        similarities.append(sim)
        
    mean_sim = statistics.mean(similarities)
    print(f"Mean Reconstruction Fidelity (Cosine Sim): {mean_sim:.4f}")
    return mean_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--R', type=int, default=5, help="Radix (R=2n+1)")
    parser.add_argument('--sigma', type=float, default=0.2, help="Noise level")
    parser.add_argument('--mode', type=str, default='wbit', choices=['wbit', 'binary'])
    args = parser.parse_args()
    
    run_wzma_experiment(args)
