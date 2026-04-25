import torch
import math

def test_champion_noise():
    print("=== VERIFYING THE CHAMPION AGAINST BASE NOISE ===")
    
    base_noises = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] 
    
    R_max = 9
    R_eff = 3 # Champion operates in Phi-Mode!
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    beta_multiplier = 1.1270
    
    # 1. Champion WZMA Matrix Setup
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
    
    # 2. Champion Beta Bias Setup
    bin_size = R_max / R_eff
    allowed_centers = [min(int((b + 0.5) * bin_size), R_max - 1) for b in range(R_eff)]
    allowed_mask = torch.zeros(R_max, dtype=torch.bool)
    allowed_mask[allowed_centers] = True
    drive_bias = torch.zeros(R_max)
    drive_bias[allowed_mask] = math.log(bin_size) * beta_multiplier
    
    batch_size = 100_000
    steps = 100 
    
    for noise_level in base_noises:
        V_m = torch.zeros(batch_size, R_max)
        success_count = 0
        
        for step in range(50 + steps): 
            noise = torch.randn(batch_size, R_max) * noise_level
            
            # CHAMPION PHYSICS: WZMA Drive + Beta Bias Pre-charge + Noise
            input_current = base_drive.unsqueeze(0) + drive_bias.unsqueeze(0) + noise
            
            dv = -(V_m / tau_leak) * dt
            V_m = torch.relu(V_m + dv + input_current * dt)
            
            if step >= 50:
                logits = V_m / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # CHAMPION PHYSICS: Phi-mode projection (Binning)
                p1 = probs[:, 0:3].sum(dim=1)
                p4 = probs[:, 3:6].sum(dim=1)
                p7 = probs[:, 6:9].sum(dim=1)
                
                projected = torch.stack([p1, p4, p7], dim=1)
                winners = torch.argmax(projected, dim=1)
                success_count += (winners == 2).sum().item()
                
        total_trials = batch_size * steps
        rate = (success_count / total_trials) * 100
        print(f"Noise {noise_level:4.2f} (SNR ~{3.0/noise_level:4.1f}) -> Champion (R_eff=3 + WZMA + Beta): {rate:6.3f}%")

if __name__ == "__main__":
    test_champion_noise()
