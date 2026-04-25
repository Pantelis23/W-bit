import torch
import math

def self_critique_test_2():
    print("=== SELF CRITIQUE 2: Testing the 'Projection' Illusion ===")
    
    # In my tests, I am projecting the 9 states down to 3 states BEFORE taking the argmax.
    # What if the OS doesn't project the probabilities, but just widens the physical read margins?
    # Does the network actually survive, or did my math projection just hide the errors?
    
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
    targeted_bias = torch.zeros(R_max)
    for c in allowed_centers:
        targeted_bias[c] = math.log(bin_size) * beta_multiplier
        
    batch_size = 100_000
    steps = 100
    
    def run_test(use_projection, name):
        V_m = torch.zeros(batch_size, R_max)
        succ = 0
        for step in range(50 + steps):
            noise = torch.randn(batch_size, R_max) * noise_level
            dv = -(V_m / tau_leak) * dt
            input_curr = champion_drive.unsqueeze(0) + targeted_bias.unsqueeze(0) + noise
            V_m = torch.relu(V_m + dv + input_curr * dt)
            if step >= 50:
                probs = torch.softmax(V_m / temperature, dim=-1)
                
                if use_projection:
                    # ML Trick: Sum probabilities before picking
                    p1 = probs[:, 0:3].sum(dim=1)
                    p4 = probs[:, 3:6].sum(dim=1)
                    p7 = probs[:, 6:9].sum(dim=1)
                    projected = torch.stack([p1, p4, p7], dim=1)
                    winners = torch.argmax(projected, dim=1)
                    succ += (winners == 2).sum().item()
                else:
                    # Physical Reality: The ADC just reads the highest voltage bin
                    # If it's 6, 7, or 8, it's considered a success because the ADC 
                    # threshold spans that entire physical range.
                    winners = torch.argmax(probs, dim=1)
                    succ += ((winners == 6) | (winners == 7) | (winners == 8)).sum().item()
                    
        rate = (succ / (batch_size * steps)) * 100
        print(f"{name:30s}: {rate:.3f}%")

    run_test(True, "Software Projection (Summing)")
    run_test(False, "Hardware ADC (Wide Margins)")

if __name__ == "__main__":
    self_critique_test_2()
