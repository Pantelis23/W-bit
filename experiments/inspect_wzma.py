import math
R_max = 9
centers = [1, 4, 7] 
A = [[math.exp(-((i - centers[k])**2) / 2.0) for k in range(3)] for i in range(R_max)]
B = [[math.exp(-((j - centers[k])**2) / 2.0) for k in range(3)] for j in range(R_max)]
wzma_matrix = [[0.0 for _ in range(R_max)] for _ in range(R_max)]
for i in range(R_max):
    for j in range(R_max):
        wzma_matrix[i][j] = sum(A[i][k] * B[j][k] for k in range(3)) * 3.0
print(f"WZMA [8][8]: {wzma_matrix[8][8]:.4f}")
print(f"WZMA [7][8]: {wzma_matrix[7][8]:.4f}")
print(f"WZMA [7][7]: {wzma_matrix[7][7]:.4f}")
