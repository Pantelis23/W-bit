"""
Quantized Plasticity Reference
Bit-exact emulation of W-bit RTL for Co-Simulation.
"""
import torch
import os

def to_int8(val):
    return max(-128, min(127, int(val)))

def to_int16(val):
    return max(-32768, min(32767, int(val)))

def simulate_rtl_update(
    w_row: list[int], # 8-bit
    x_val: int,       # 16-bit
    y_vec: list[int], # 32-bit (accum)
    lr_shift: int = 20
) -> list[int]:
    w_out = []
    for j, w in enumerate(w_row):
        y = y_vec[j]
        prod = x_val * y
        delta = prod >> lr_shift
        w_sum = w + delta
        w_new = max(-128, min(127, w_sum))
        w_out.append(w_new)
    return w_out

def generate_test_vectors(out_dir="test_vectors"):
    os.makedirs(out_dir, exist_ok=True)
    
    COLS = 256
    
    # 1. Initial State (Weights)
    w_init = [to_int8(i % 20) for i in range(COLS)]
    
    # 2. Inputs
    # x = 1000, y = 20000 -> prod = 20M >> 20 = 19
    x_val = 1000
    y_vec = [20000] * COLS 
    
    # 3. Compute Golden Output
    w_golden = simulate_rtl_update(w_init, x_val, y_vec, lr_shift=20)
    
    # 4. Save Hex
    with open(f"{out_dir}/weights_init.hex", "w") as f:
        for w in w_init:
            f.write(f"{w & 0xFF:02X}\n")
            
    with open(f"{out_dir}/inputs.hex", "w") as f:
        f.write(f"{x_val & 0xFFFF:04X}\n") # x
        for y in y_vec:
            f.write(f"{y & 0xFFFFFFFF:08X}\n") # y
            
    with open(f"{out_dir}/weights_golden.hex", "w") as f:
        for w in w_golden:
            f.write(f"{w & 0xFF:02X}\n")
            
    print(f"Generated test vectors in {out_dir}")
    print(f"Sample: W[0]={w_init[0]} -> {w_golden[0]} (Delta +{w_golden[0]-w_init[0]})")

if __name__ == "__main__":
    generate_test_vectors()