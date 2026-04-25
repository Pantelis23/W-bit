import torch
import math

def test_complex_noise():
    print("=== TESTING BEYOND GAUSSIAN CATASTROPHE ===")
    
    R_max = 9
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    beta_multiplier = 1.1270
    
    centers = [1, 4, 7]
    A = torch.zeros(R_max, 3)
    B = torch.zeros(R_max, 3)
    for k in range(3):
        for i in range(R_max):
            A[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
            B[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
    unscaled_88 = sum(A[8, k].item() * B[8, k].item() for k in range(3))
    wzma_scale = 3.0 / unscaled_88
    wzma_matrix = torch.matmul(A, B.T) * wzma_scale
    analog_drive = wzma_matrix[:, 8]
    
    bin_size = R_max / 3
    allowed_centers = [min(int((b + 0.5) * bin_size), R_max - 1) for b in range(3)]
    targeted_bias = torch.zeros(R_max)
    for c in allowed_centers:
        targeted_bias[c] = math.log(bin_size) * beta_multiplier
        
    batch_size = 5_000
    steps = 1000
    
    def run_noise_scenario(name, noise_fn):
        V_m = torch.zeros(batch_size, R_max)
        succ = 0
        for step in range(50 + steps):
            # Let the custom noise function generate the noise for this step
            noise = noise_fn(batch_size, R_max, step)
            
            dv = -(V_m / tau_leak) * dt
            input_curr = analog_drive.unsqueeze(0) + targeted_bias.unsqueeze(0) + noise
            V_m = torch.relu(V_m + dv + input_curr * dt)
            
            if step >= 50:
                probs = torch.softmax(V_m / temperature, dim=-1)
                winners = torch.argmax(probs, dim=1)
                succ += ((winners == 6) | (winners == 7) | (winners == 8)).sum().item()
                
        total = batch_size * steps
        rate = (succ / total) * 100
        print(f"{name:30s} | Accuracy: {rate:6.3f}%")

    # 1. Baseline standard Gaussian
    run_noise_scenario("Standard Gaussian (5.0)", 
                       lambda b, r, s: torch.randn(b, r) * 5.0)
                       
    # 2. Burst Noise (Random massive spikes of 20.0 every ~100 steps)
    def burst_noise(b, r, s):
        base = torch.randn(b, r) * 2.0 # Lower base noise
        burst_mask = (torch.rand(b, 1) < 0.01).float() # 1% chance of burst
        bursts = torch.randn(b, r) * 20.0 * burst_mask
        return base + bursts
    run_noise_scenario("Burst Noise (Spikes of 20.0)", burst_noise)

    # 3. Slow Drift (Sine wave drifting the baseline over time)
    def slow_drift(b, r, s):
        base = torch.randn(b, r) * 2.0
        drift = math.sin(s / 20.0) * 5.0 # Slow wave peaking at 5.0
        return base + drift
    run_noise_scenario("Slow Correlated Drift (5.0)", slow_drift)
    
    # 4. Asymmetrical Offset Drift (Comparators drifting downwards)
    def offset_drift(b, r, s):
        base = torch.randn(b, r) * 2.0
        # Constant negative pull, simulating a hardware comparator leak/bias
        return base - 4.0 
    run_noise_scenario("Asymmetrical Offset (-4.0)", offset_drift)

if __name__ == "__main__":
    test_complex_noise()
