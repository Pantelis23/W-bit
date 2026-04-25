"""
Plasticity Support for WZMA
Implements online-updating low-rank matrices (Hebbian Learning).
Based on AtlasLM/src/hybrid_plasticity/plastic_linear.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PlasticLowRankBank(nn.Module):
    """
    A single 'Plastic' weight bank for WZMA.
    Updates U and V online using Hebbian rules.
    """
    def __init__(self, d_in, d_out, rank=16, lr=0.01, decay=0.99):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.lr = lr
        self.decay = decay
        
        # Fast weights - Initialize with small noise to break symmetry/silence
        self.register_buffer('U', torch.randn(d_out, rank) * 0.01)
        self.register_buffer('V', torch.randn(d_in, rank) * 0.01)
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x: [B, S, d_in]
        
        # CLONE weights to prevent in-place modification error during backward
        # This simulates the hardware reading the weights at time t.
        V_curr = self.V.clone()
        U_curr = self.U.clone()
        
        h = x @ V_curr # [B, S, r]
        out = h @ U_curr.T # [B, S, d_out]
        return out

    @torch.no_grad()
    def update(self, x_pre, y_grad_proxy):
        """
        Hebbian update step.
        """
        self.step_count += 1
        
        # Decay
        self.U.mul_(self.decay)
        self.V.mul_(self.decay)
        
        # Random Projection Hebbian
        r_idx = torch.randint(0, self.rank, (1,)).item()
        
        pre_act = x_pre.mean(dim=(0, 1)) # [d_in]
        post_act = y_grad_proxy.mean(dim=(0, 1)) # [d_out]
        
        pre_act = F.normalize(pre_act, dim=0)
        post_act = F.normalize(post_act, dim=0)
        
        self.U[:, r_idx] += self.lr * post_act
        self.V[:, r_idx] += self.lr * pre_act
        
        self.U.clamp_(-1.0, 1.0)
        self.V.clamp_(-1.0, 1.0)

    def reset(self):
        self.U.normal_(0, 0.01)
        self.V.normal_(0, 0.01)
        self.step_count.zero_()
