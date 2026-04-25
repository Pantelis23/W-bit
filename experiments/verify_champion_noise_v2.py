import torch
import math

def test_champion_noise():
    print("=== VERIFYING THE BASELINE (R=9) VS CHAMPION (R=3) ===")
    
    # We test the exact crossover point where the OS MUST switch modes.
    noises = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] 
    
    R_max = 9
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    beta_multiplier = 1.1270
    
    # --- 1. R=9 High-Precision Setup (The Fragile Baseline) ---
    base_matrix = torch.zeros(R_max, R_max)
    for i in range(R_max): base_matrix[i, i] = 3.0 
    base_drive_r9 = base_matrix[:, 8]
    
    # --- 2. R=3 Champion Setup (WZMA + Beta) ---
    centers = [1, 4, 7]
    A = torch.zeros(R_max, 3)
    B = torch.zeros(R_max, 3)
    for k in range(3):
        for i in range(R_max):
            A[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
            B[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
    unscaled_88 = sum(A[8, k].item() * B[8, k].item() for k in range(3))
    wzma_matrix = torch.matmul(A, B.T) * (3.0 / unscaled_88)
    champion_drive = wzma_matrix[:, 8]
    
    bin_size = R_max / 3
    allowed_centers = [min(int((b + 0.5) * bin_size), R_max - 1) for b in range(3)]
    allowed_mask = torch.zeros(R_max, dtype=torch.bool)
    allowed_mask[allowed_centers] = True
    champion_bias = torch.zeros(R_max)
    champion_bias[allowed_mask] = math.log(bin_size) * beta_multiplier
    
    batch_size = 100_000
    steps = 100 
    
    print(f"{'Noise':<6} | {'SNR':<5} || {'R=9 Baseline':<15} | {'Champion (R=3)':<15}")
    print("-" * 55)
    
    for noise_level in noises:
        
        # Test R=9
        V_m_r9 = torch.zeros(batch_size, R_max)
        succ_r9 = 0
        for step in range(50 + steps): 
            noise = torch.randn(batch_size, R_max) * noise_level
            dv = -(V_m_r9 / tau_leak) * dt
            V_m_r9 = torch.relu(V_m_r9 + dv + (base_drive_r9.unsqueeze(0) + noise) * dt)
            if step >= 50:
                winners = torch.argmax(torch.softmax(V_m_r9 / temperature, dim=-1), dim=1)
                succ_r9 += (winners == 8).sum().item()
                
        # Test Champion (R=3)
        V_m_champ = torch.zeros(batch_size, R_max)
        succ_champ = 0
        for step in range(50 + steps): 
            noise = torch.randn(batch_size, R_max) * noise_level
            dv = -(V_m_champ / tau_leak) * dt
            input_curr = champion_drive.unsqueeze(0) + champion_bias.unsqueeze(0) + noise
            V_m_champ = torch.relu(V_m_champ + dv + input_curr * dt)
            if step >= 50:
                probs = torch.softmax(V_m_champ / temperature, dim=-1)
                projected = torch.stack([probs[:, 0:3].sum(dim=1), probs[:, 3:6].sum(dim=1), probs[:, 6:9].sum(dim=1)], dim=1)
                winners = torch.argmax(projected, dim=1)
                succ_champ += (winners == 2).sum().item()
                
        trials = batch_size * steps
        r9_rate = (succ_r9 / trials) * 100
        champ_rate = (succ_champ / trials) * 100
        
        print(f"{noise_level:<6.2f} | {3.0/noise_level:<5.1f} || {r9_rate:>14.3f}% | {champ_rate:>14.3f}%")

if __name__ == "__main__":
    test_champion_noise()
