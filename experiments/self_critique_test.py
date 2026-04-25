import torch
import math

def self_critique_test():
    print("=== SELF CRITIQUE: Testing the 'Beta Bias' Assumption ===")
    
    # In my previous tests, I hard-coded the Beta Bias to only apply to the 
    # specific 'allowed_centers' (States 1, 4, 7). 
    # I need to see what happens if the bias is applied globally, or if it's 
    # just an artificial crutch I built into the test.
    
    R_max = 9
    R_eff = 3
    dt = 1.0
    temperature = 0.5
    tau_leak = 5.0
    beta_multiplier = 1.1270
    noise_level = 5.0
    
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
    
    bin_size = R_max / R_eff
    allowed_centers = [min(int((b + 0.5) * bin_size), R_max - 1) for b in range(R_eff)]
    
    # 1. Targeted Bias (What I did before)
    targeted_bias = torch.zeros(R_max)
    for c in allowed_centers:
        targeted_bias[c] = math.log(bin_size) * beta_multiplier
        
    # 2. Global Bias (Adding energy to everything)
    global_bias = torch.full((R_max,), math.log(bin_size) * beta_multiplier)
    
    batch_size = 100_000
    steps = 100
    
    def run_test(bias_tensor, name):
        V_m = torch.zeros(batch_size, R_max)
        succ = 0
        for step in range(50 + steps):
            noise = torch.randn(batch_size, R_max) * noise_level
            dv = -(V_m / tau_leak) * dt
            input_curr = champion_drive.unsqueeze(0) + bias_tensor.unsqueeze(0) + noise
            V_m = torch.relu(V_m + dv + input_curr * dt)
            if step >= 50:
                probs = torch.softmax(V_m / temperature, dim=-1)
                projected = torch.stack([probs[:, 0:3].sum(dim=1), probs[:, 3:6].sum(dim=1), probs[:, 6:9].sum(dim=1)], dim=1)
                winners = torch.argmax(projected, dim=1)
                succ += (winners == 2).sum().item()
        rate = (succ / (batch_size * steps)) * 100
        print(f"{name:20s}: {rate:.3f}%")

    run_test(torch.zeros(R_max), "No Bias (Baseline)")
    run_test(targeted_bias, "Targeted Beta Bias")
    run_test(global_bias, "Global Beta Bias")

if __name__ == "__main__":
    self_critique_test()
