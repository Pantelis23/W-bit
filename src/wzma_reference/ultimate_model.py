"""
WZMA Ultimate: The Convergence Architecture
Combines:
1. WZMA (Factorized Linear) - Efficient Basics
2. Plasticity (Hebbian) - Self-Adjustment
3. S4 (State Space) - Infinite Context
4. Hyperdimensional - Noise Robustness
5. Grouped Query Attention - Cache Efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from .plasticity import PlasticLowRankBank
from .model import WZMALinear, WZMAConfig # Base WZMA

@dataclass
class UltimateConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2     # Grouped Query Attention (3 queries per key)
    d_state: int = 16       # S4 State dimension
    dropout: float = 0.1
    # Plasticity
    enable_plasticity: bool = True
    plastic_lr: float = 0.01

# --- 1. Hyperdimensional Projection (Noise Robustness) ---
class HyperdimLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # Bipolar weights (+1/-1) fixed or learnable
        # For HW efficiency, we assume fixed random projection, but learned scale
        self.proj = nn.Linear(d_in, d_out, bias=False)
        with torch.no_grad():
            self.proj.weight.data.sign_() # Binarize init
            
    def forward(self, x):
        # Bipolar activation function (Softsign or Tanh approximation)
        return torch.tanh(self.proj(x))

# --- 2. S4 Layer (Simplified for Reference) ---
# Structured State Space Sequence Model (Hippo)
class S4Layer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # A, B, C matrices (Simplified Diagonal S4)
        # Discretized evolution parameters
        
        # STABILITY FIX: Initialize A_log to ensure stable A (-1 < A < 0)
        # exp(A_log) should be small (0.01 to 0.1)
        # A_log range: log(0.01)=-4.6 to log(0.1)=-2.3
        self.A_log = nn.Parameter(torch.empty(d_model, d_state))
        nn.init.uniform_(self.A_log, -4.0, -2.0)
        
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x):
        # x: [Batch, Seq, d_model]
        # Recurrent mode simulation (O(N))
        
        batch, seq, dim = x.shape
        h = torch.zeros(batch, dim, self.d_state, device=x.device)
        outputs = []
        
        # Discretize A (approximation)
        # A must be negative for stability
        # Use sigmoid to strictly bound if needed, but exp(log) is standard S4
        A = -torch.exp(self.A_log) # Ensure stability
        
        for t in range(seq):
            u = x[:, t, :] # [B, D]
            
            # State Update: h' = h + h*A + u*B
            # h' = h(1+A) + uB
            # If 1+A > 1 or < -1, explode.
            # A is negative. Ensure A > -2.
            # Our init ensures A approx -0.1. So 1+A = 0.9. Stable.
            
            term1 = h * A.unsqueeze(0)
            term2 = u.unsqueeze(2) * self.B.unsqueeze(0)
            h = h + term1 + term2
            
            # Output: y = Ch + Du
            y = torch.sum(h * self.C.unsqueeze(0), dim=2) + u * self.D.unsqueeze(0)
            outputs.append(y)
            
        return torch.stack(outputs, dim=1)

# --- 3. Grouped Query Attention (Efficiency) ---
class GQA(nn.Module):
    def __init__(self, config: UltimateConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # Create WZMA Config subset
        w_conf = WZMAConfig(
            d_model=config.d_model,
            n_banks=4, # Default banks for projections
            rank=32,
            enable_plasticity=False # Projections are static, FFN is plastic
        )
        
        # Projections (Using WZMA Linear for efficiency!)
        # Q is full heads, K/V are fewer heads
        self.q_proj = WZMALinear(config.d_model, config.d_model, w_conf)
        self.k_proj = WZMALinear(config.d_model, self.n_kv_heads * self.head_dim, w_conf)
        self.v_proj = WZMALinear(config.d_model, self.n_kv_heads * self.head_dim, w_conf)
        self.o_proj = nn.Linear(config.d_model, config.d_model)
        
    def forward(self, x, mask=None):
        B, S, D = x.shape
        
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat K/V for GQA
        # k: [B, n_kv, S, d] -> [B, n_heads, S, d]
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Attention
        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.o_proj(out)

# --- 4. Plastic Feed Forward (Self-Adjustment) ---
class PlasticFFN(nn.Module):
    def __init__(self, config: UltimateConfig):
        super().__init__()
        # WZMA with Plasticity enabled on Bank 0
        wzma_conf = WZMAConfig(
            d_model=config.d_model, 
            d_ff=config.d_model*4,
            enable_plasticity=config.enable_plasticity,
            plastic_lr=config.plastic_lr
        )
        # Up projection
        self.up = WZMALinear(config.d_model, config.d_model*4, wzma_conf)
        self.act = nn.GELU()
        self.down = nn.Linear(config.d_model*4, config.d_model)
        
    def forward(self, x):
        return self.down(self.act(self.up(x)))

# --- ULTIMATE MODEL ---
class WZMAUltimate(nn.Module):
    def __init__(self, config: UltimateConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Hyperdimensional Pre-processing (Robustness)
        self.hyper = HyperdimLayer(config.d_model, config.d_model)
        
        # Stack
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                's4': S4Layer(config.d_model, config.d_state),
                'attn': GQA(config),
                'ffn': PlasticFFN(config),
                'ln1': nn.LayerNorm(config.d_model),
                'ln2': nn.LayerNorm(config.d_model),
                'ln3': nn.LayerNorm(config.d_model)
            })
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        
    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        
        # 1. Hyperdimensional Projection
        # Adds "width" to the signal (robustness)
        x = x + self.hyper(x)
        
        for layer in self.layers:
            # 2. S4 (Long Context Mixing)
            res = x
            x = layer['ln1'](x)
            x = res + layer['s4'](x)
            
            # 3. Attention (Short Context Reasoning)
            res = x
            x = layer['ln2'](x)
            x = res + layer['attn'](x, mask)
            
            # 4. Plastic FFN (Knowledge & Adjustment)
            res = x
            x = layer['ln3'](x)
            x = res + layer['ffn'](x)
            
        return self.ln_f(x)
