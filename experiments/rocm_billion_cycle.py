import torch
import math
import time

def run_billion():
    print("=== ONE BILLION CYCLE STRESS TEST (ROCm Accelerated) ===")
    
    # Check ROCm / AMD GPU availability
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: PyTorch cannot see the AMD GPU.")
        print("Is the ROCm stack installed and exported properly? (HIP_VISIBLE_DEVICES)")
        return
        
    device = torch.device('cuda')
    print(f"Device acquired: {torch.cuda.get_device_name(0)}")
    
    noise_level = 5.0
    R_max = 9
    R_eff = 3
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    
    # WZMA Matrix (Pure Physics)
    centers = [1, 4, 7]
    A = torch.zeros(R_max, 3, device=device)
    B = torch.zeros(R_max, 3, device=device)
    for k in range(3):
        for i in range(R_max):
            A[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
            B[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
            
    unscaled_88 = sum(A[8, k].item() * B[8, k].item() for k in range(3))
    wzma_scale = 3.0 / unscaled_88
    wzma_matrix = torch.matmul(A, B.T) * wzma_scale
    
    # We broadcast State 8. The drive is the 8th column of the WZMA matrix.
    base_drive = wzma_matrix[:, 8]
    
    # We are testing the FINAL, stripped-down Level 3 physics:
    # 1. WZMA Smooth Landscape
    # 2. Tau-bit Integration (Leak over time)
    # 3. Phi-Mode binning (R_eff = 3)
    
    batch_size = 200_000 # Maximize parallel ROCm throughput
    steps = 5_000 # 200k * 5k = 1 Billion Total Trials
    
    start_time = time.time()
    
    # V_m tracks the analog voltage across the R_max bins for all 200,000 parallel cells
    V_m = torch.zeros(batch_size, R_max, device=device)
    success_count = 0
    
    print(f"Executing 1,000,000,000 integrations. Batch: {batch_size}, Steps: {steps}")
    
    for step in range(steps):
        # 1. Inject pure Gaussian thermal noise
        noise = torch.randn(batch_size, R_max, device=device) * noise_level
        
        # 2. Add the clean WZMA signal to the noise
        input_current = base_drive.unsqueeze(0) + noise
        
        # 3. Apply the Tau-bit physics (Leak + Integration)
        dv = -(V_m / tau_leak) * dt
        V_m = torch.relu(V_m + dv + input_current * dt)
        
        # We start counting successes after a 50-step "warmup" to let the voltages settle
        if step >= 50:
            # Settle into probabilities based on physical heat (temperature)
            logits = V_m / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Level 3 Phi-Mode: Collapse the 9-dimensional physics into 3 robust logic bins
            p1 = probs[:, 0:3].sum(dim=1)
            p4 = probs[:, 3:6].sum(dim=1)
            p7 = probs[:, 6:9].sum(dim=1)
            
            projected = torch.stack([p1, p4, p7], dim=1)
            
            # The logic gate "snaps" to the highest bin
            winners = torch.argmax(projected, dim=1)
            
            # The input signal was State 8. The correct Phi-mode bin for 8 is the top bin (index 2, centered at 7)
            success_count += (winners == 2).sum().item()
            
            if step % 1000 == 0:
                print(f"  Step {step}/5000 completed...")
            
    total_valid_trials = batch_size * (steps - 50)
    rate = (success_count / total_valid_trials) * 100
    elapsed = time.time() - start_time
    
    print("\n=== FINAL RESULTS ===")
    print(f"Total Integrations: {total_valid_trials:,}")
    print(f"Successful Signals: {success_count:,}")
    print(f"Survival Rate:      {rate:.4f}%")
    print(f"Wall-clock time:    {elapsed:.1f} seconds")
    print(f"Throughput:         {(total_valid_trials / elapsed) / 1e6:.1f} Million ops/sec")

if __name__ == "__main__":
    run_billion()
