import torch
import time

def estimate_cpu():
    print("=== ESTIMATING 1 BILLION CYCLES ON CPU ===")
    
    # We run a small batch (100,000 total cycles) to measure CPU throughput
    # and then extrapolate to 1,000,000,000.
    
    batch_size = 10_000
    steps = 10 
    
    R_max = 9
    noise_level = 5.0
    dt = 1.0
    tau_leak = 5.0
    temperature = 0.5
    
    base_drive = torch.zeros(R_max) # Dummy data for speed test
    V_m = torch.zeros(batch_size, R_max)
    
    # Warmup to clear Python overhead
    for _ in range(2):
        noise = torch.randn(batch_size, R_max) * noise_level
        dv = -(V_m / tau_leak) * dt
        V_m = torch.relu(V_m + dv + (base_drive + noise) * dt)
    
    # Timing run
    start_time = time.time()
    
    for step in range(steps):
        noise = torch.randn(batch_size, R_max) * noise_level
        dv = -(V_m / tau_leak) * dt
        V_m = torch.relu(V_m + dv + (base_drive + noise) * dt)
        
        logits = V_m / temperature
        probs = torch.softmax(logits, dim=-1)
        
        p1 = probs[:, 0:3].sum(dim=1)
        p4 = probs[:, 3:6].sum(dim=1)
        p7 = probs[:, 6:9].sum(dim=1)
        
        projected = torch.stack([p1, p4, p7], dim=1)
        winners = torch.argmax(projected, dim=1)
        
    elapsed = time.time() - start_time
    
    total_cycles = batch_size * steps
    cycles_per_second = total_cycles / elapsed
    
    billion = 1_000_000_000
    seconds_needed = billion / cycles_per_second
    minutes_needed = seconds_needed / 60
    hours_needed = minutes_needed / 60
    
    print(f"CPU Throughput: {cycles_per_second:,.0f} cycles/sec")
    print(f"Time to 1 Billion: {seconds_needed:,.0f} seconds")
    print(f"                   {minutes_needed:.1f} minutes")
    if hours_needed > 1.0:
        print(f"                   {hours_needed:.2f} hours")

if __name__ == "__main__":
    estimate_cpu()
