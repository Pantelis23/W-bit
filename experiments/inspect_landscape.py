import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def print_landscape():
    print("=== INSPECTING THE ENERGY LANDSCAPE ===\n")
    R_max = 9
    wzma_scale = 13.6420
    centers = [1, 4, 7] 
    
    # 1. Generate the raw WZMA interaction weights (The base "slope")
    A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
    B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
    
    wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
    for i in range(R_max):
        for j in range(R_max):
            dot = sum(A[i][k] * B[j][k] for k in range(3))
            wzma_matrix[i][j] = dot * wzma_scale
            
    print("Base WZMA Energy Landscape (when receiving perfect State 7):")
    base_drive = [wzma_matrix[r][7] for r in range(R_max)]
    for r, drive in enumerate(base_drive):
        marker = "<- CENTER" if r in centers else ""
        print(f"  State {r}: {drive:6.2f}  {marker}")
        
    print("\n--- Applying H-Neuron Penalty (-9.18 to boundaries) ---")
    penalty_drive = list(base_drive)
    for r in range(R_max):
        if r not in centers:
            penalty_drive[r] -= 9.1883
            
    for r, drive in enumerate(penalty_drive):
        marker = "<- CENTER" if r in centers else ""
        print(f"  State {r}: {drive:6.2f}  {marker}")
        
    print("\n--- Applying Softmax (Temperature = 1.0) ---")
    
    def softmax(x, temp=1.0):
        m = max(x)
        exps = [math.exp((val - m) / temp) for val in x]
        s = sum(exps)
        return [e / s for e in exps]
        
    prob_base = softmax(base_drive)
    prob_penalty = softmax(penalty_drive)
    
    print("Probabilities WITHOUT penalty (Smooth):")
    for r, p in enumerate(prob_base):
        print(f"  State {r}: {p*100:5.1f}%")
        
    print("\nProbabilities WITH penalty (Shattered):")
    for r, p in enumerate(prob_penalty):
        print(f"  State {r}: {p*100:5.1f}%")

if __name__ == "__main__":
    print_landscape()
