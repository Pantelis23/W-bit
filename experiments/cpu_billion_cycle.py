import torch
import math
import time

def run_billion_cpu():
    print("=== ONE BILLION CYCLE STRESS TEST (CPU) ===")
    
    noise_level = 5.0
    R_max = 9
    R_eff = 3
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    tau_h_leak = 1.0
    beta_multiplier = 1.1270
    
    # WZMA Matrix
    centers = [1, 4, 7]
    A = torch.zeros(R_max, 3)
    B = torch.zeros(R_max, 3)
    for k in range(3):
        for i in range(R_max):
            A[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
            B[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
            
    unscaled_88 = sum(A[8, k] * B[8, k] for k in range(3))
    wzma_scale = 3.0 / unscaled_88
    wzma_matrix = torch.matmul(A, B.T) * wzma_scale
    base_drive = wzma_matrix[:, 8]
    
    bin_size = R_max / R_eff
    allowed_centers = [min(int((b + 0.5) * bin_size), R_max - 1) for b in range(R_eff)]
    allowed_mask = torch.zeros(R_max, dtype=torch.bool)
    allowed_mask[allowed_centers] = True
    
    configs = [
        ("Baseline (WZMA only)", False, False),
        ("With True Beta Bias", True, False),
        ("With H-Neuron Leak", False, True),
        ("Beta + H-Neuron", True, True)
    ]
    
    # We will do 100 million per config to total 400 million, 
    # since 1 Billion per config would take 4.8 hours.
    batch_size = 100_000
    steps = 1_000 
    
    print(f"Running {batch_size * steps:,} cycles per configuration...")
    
    for name, use_beta, use_h in configs:
        start_time = time.time()
        
        tau = torch.full((R_max,), tau_leak)
        if use_h:
            tau[~allowed_mask] = tau_h_leak
            
        drive_bias = torch.zeros(R_max)
        if use_beta:
            drive_bias[allowed_mask] = math.log(bin_size) * beta_multiplier
            
        V_m = torch.zeros(batch_size, R_max)
        success_count = 0
        
        for step in range(50 + steps): 
            noise = torch.randn(batch_size, R_max) * noise_level
            input_current = base_drive.unsqueeze(0) + drive_bias.unsqueeze(0) + noise
            dv = -(V_m / tau.unsqueeze(0)) * dt
            V_m = torch.relu(V_m + dv + input_current * dt)
            
            if step >= 50:
                logits = V_m / temperature
                probs = torch.softmax(logits, dim=-1)
                
                p1 = probs[:, 0:3].sum(dim=1)
                p4 = probs[:, 3:6].sum(dim=1)
                p7 = probs[:, 6:9].sum(dim=1)
                
                projected = torch.stack([p1, p4, p7], dim=1)
                winners = torch.argmax(projected, dim=1)
                success_count += (winners == 2).sum().item()
                
        total_trials = batch_size * steps
        rate = (success_count / total_trials) * 100
        elapsed = time.time() - start_time
        print(f"  {name:25s} : {success_count:11,d} / {total_trials:,} ({rate:6.3f}%) | Time: {elapsed:.1f}s")

if __name__ == "__main__":
    run_billion_cpu()
