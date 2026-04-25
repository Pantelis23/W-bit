import torch
import math

def test_base_noise():
    print("=== ESTIMATING NORMAL BASELINE NOISE (Corrected R=9 logic) ===")
    
    # We must use a full-rank (dense) matrix for R=9 high-precision mode,
    # because the WZMA matrix we used earlier is literally mathematically
    # incapable of holding State 8. WZMA collapses everything into States 1, 4, 7!
    
    base_noises = [0.05, 0.1, 0.2, 0.5, 1.0] 
    
    R_max = 9
    R_eff = 9 # Testing in High-Precision Mode
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    
    # The Baseline Diagonal Matrix (Rigid, but allows all 9 states)
    base_matrix = torch.zeros(R_max, R_max)
    for i in range(R_max):
        base_matrix[i, i] = 3.0 # Peak Signal is 3.0
        
    base_drive = base_matrix[:, 8] # Broadcasting State 8
    
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
                
                # In R=9 high-precision mode, we just take argmax directly.
                winners = torch.argmax(probs, dim=1)
                success_count += (winners == 8).sum().item()
                
        total_trials = batch_size * steps
        rate = (success_count / total_trials) * 100
        print(f"Noise {noise_level:4.2f} (SNR ~{3.0/noise_level:4.1f}) -> High-Precision (R=9) Survival: {rate:6.3f}%")

if __name__ == "__main__":
    test_base_noise()
