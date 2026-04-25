import torch
import math

def test_binary_noise():
    print("=== SCORING BINARY LOGIC (R=2) UNDER CATASTROPHIC NOISE ===")
    print("Testing pure physics: Binary Matrix + Tau Integration")
    print("Noise Level: 5.0 (Catastrophic) | Signal Peak: 3.0\n")
    
    R_max = 2 # Binary (0 or 1)
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    noise_level = 5.0
    
    # 1. Binary Diagonal Matrix (The ultimate rigid landscape)
    # State 0 prefers 0, State 1 prefers 1.
    base_matrix = torch.zeros(R_max, R_max)
    base_matrix[0, 0] = 3.0
    base_matrix[1, 1] = 3.0
    
    # Broadcasting State 1 (The intended signal)
    binary_drive = base_matrix[:, 1]
    
    batch_size = 100_000
    steps = 1000
    
    V_m = torch.zeros(batch_size, R_max)
    succ = 0
    
    for step in range(50 + steps):
        noise = torch.randn(batch_size, R_max) * noise_level
        
        # Tau-bit Integration
        dv = -(V_m / tau_leak) * dt
        input_curr = binary_drive.unsqueeze(0) + noise
        V_m = torch.relu(V_m + dv + input_curr * dt)
        
        if step >= 50:
            probs = torch.softmax(V_m / temperature, dim=-1)
            winners = torch.argmax(probs, dim=1)
            
            # Hardware ADC: If it's State 1, it's a success
            succ += (winners == 1).sum().item()
                
    total = batch_size * steps
    rate = (succ / total) * 100
    print(f"Final Survival Score: {succ:,} / {total:,}")
    print(f"Accuracy:             {rate:.4f}%")

if __name__ == "__main__":
    test_binary_noise()
