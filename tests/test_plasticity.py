"""
Verify Plasticity in WZMA
Tests that the plastic bank updates its weights online.
"""
import pytest
import torch
import os
import sys

# Add W-bit root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.wzma_reference.model import WZMAEncoder, WZMAConfig

def test_online_plasticity():
    # 1. Config with plasticity
    config = WZMAConfig(
        d_model=64, n_layers=1, n_heads=2, 
        enable_plasticity=True, plastic_lr=0.1
    )
    model = WZMAEncoder(config)
    model.train() # Plasticity requires training mode
    
    # 2. Input
    input_ids = torch.randint(0, 100, (1, 10))
    
    # 3. Snapshot plastic weights
    layer = model.layers[0].ff1 # First plastic layer
    U_init = layer.plastic_bank.U.clone()
    V_init = layer.plastic_bank.V.clone()
    
    # 4. Forward Pass (triggers update)
    _ = model(input_ids)
    
    # 5. Check Update
    U_post = layer.plastic_bank.U
    V_post = layer.plastic_bank.V
    
    # Weights should have changed
    assert not torch.allclose(U_init, U_post), "Plastic U did not update!"
    assert not torch.allclose(V_init, V_post), "Plastic V did not update!"
    
    # Verify step count
    assert layer.plastic_bank.step_count == 1
    
    print("Plasticity verified: Weights updated online.")

if __name__ == "__main__":
    test_online_plasticity()
