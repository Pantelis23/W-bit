import torch
import math

def test_base_noise():
    print("=== ESTIMATING NORMAL BASELINE NOISE ===")
    
    # We want to test the realistic noise a physical analog memristor (like Morphium-E)
    # would experience at room temperature, without catastrophic thermal spikes.
    
    # In standard analog computing, a SNR (Signal-to-Noise Ratio) of 10dB to 20dB is common.
    # If our peak signal is 3.0, a realistic baseline noise floor (Gaussian standard deviation)
    # would be roughly 10% to 20% of the signal magnitude.
    
    base_noises = [0.1, 0.3, 0.5, 1.0] # 3%, 10%, 16%, and 33% of the signal
    
    R_max = 9
    R_eff = 9 # Testing in High-Precision Mode!
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    
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
    
    batch_size = 100_000
    steps = 100 
    
    for noise_level in base_noises:
        V_m = torch.zeros(batch_size, R_max)
        success_count = 0
        
        for step in range(50 + steps): 
            noise = torch.randn(batch_size, R_max) * noise_level
            input_current = base_drive.unsqueeze(0) + noise
            dv = -(V_m / tau_leak) * dt
            V_m = torch.relu(V_m + dv + input_current * dt)
            
            if step >= 50:
                logits = V_m / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # In R=9 high-precision mode, we don't project. We just take argmax directly.
                winners = torch.argmax(probs, dim=1)
                success_count += (winners == 8).sum().item()
                
        total_trials = batch_size * steps
        rate = (success_count / total_trials) * 100
        print(f"Noise {noise_level:3.1f} (SNR ~{3.0/noise_level:4.1f}) -> High-Precision (R=9) Survival: {rate:6.3f}%")

if __name__ == "__main__":
    test_base_noise()
