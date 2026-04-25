import torch
import math

def score_true_analog():
    print("=== SCORING THE TRUE ANALOG ARCHITECTURE ===")
    print("Testing pure physics: WZMA Slopes + Beta Pre-Charge + Wide ADC Margins + Tau Integration")
    print("Noise Level: 5.0 (Catastrophic) | Signal Peak: 3.0\n")
    
    R_max = 9
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    beta_multiplier = 1.1270
    noise_level = 5.0
    
    # 1. Physical WZMA Slopes
    centers = [1, 4, 7]
    A = torch.zeros(R_max, 3)
    B = torch.zeros(R_max, 3)
    for k in range(3):
        for i in range(R_max):
            A[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
            B[i, k] = math.exp(-((i - centers[k])**2) / 2.0)
    unscaled_88 = sum(A[8, k].item() * B[8, k].item() for k in range(3))
    wzma_matrix = torch.matmul(A, B.T) * (3.0 / unscaled_88)
    analog_drive = wzma_matrix[:, 8]
    
    # 2. Targeted Beta Pre-Charge
    bin_size = R_max / 3
    allowed_centers = [min(int((b + 0.5) * bin_size), R_max - 1) for b in range(3)]
    targeted_bias = torch.zeros(R_max)
    for c in allowed_centers:
        targeted_bias[c] = math.log(bin_size) * beta_multiplier
        
    batch_size = 100_000
    steps = 1000
    
    V_m = torch.zeros(batch_size, R_max)
    succ = 0
    
    for step in range(50 + steps):
        noise = torch.randn(batch_size, R_max) * noise_level
        
        # 3. Tau-bit Integration
        dv = -(V_m / tau_leak) * dt
        input_curr = analog_drive.unsqueeze(0) + targeted_bias.unsqueeze(0) + noise
        V_m = torch.relu(V_m + dv + input_curr * dt)
        
        if step >= 50:
            probs = torch.softmax(V_m / temperature, dim=-1)
            winners = torch.argmax(probs, dim=1)
            
            # 4. Hardware ADC Wide Margins (R_eff = 3)
            # The top bucket physically spans voltages 6, 7, and 8.
            succ += ((winners == 6) | (winners == 7) | (winners == 8)).sum().item()
                
    total = batch_size * steps
    rate = (succ / total) * 100
    print(f"Final Survival Score: {succ:,} / {total:,}")
    print(f"Accuracy:             {rate:.4f}%")

if __name__ == "__main__":
    score_true_analog()
