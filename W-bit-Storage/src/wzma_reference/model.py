"""
WZMA Encoder Model
Implements factorized weight banks with input-dependent gating.
Now supports Online Plasticity (Hebbian Learning) on the first bank.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dataclasses import dataclass
from .plasticity import PlasticLowRankBank

@dataclass
class WZMAConfig:
    vocab_size: int = 8192
    max_seq_len: int = 128
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    # WZMA specifics
    n_banks: int = 4        # Number of weight banks
    rank: int = 32          # Rank for factorized weights
    dropout: float = 0.1
    # Plasticity
    enable_plasticity: bool = False
    plastic_lr: float = 0.01

class WZMALinear(nn.Module):
    """
    Factorized Linear Layer with Soft Gating (WZMA-style).
    Optionally replaces Bank 0 with a Plastic (Online-Updating) Bank.
    """
    def __init__(self, d_in, d_out, config: WZMAConfig):
        super().__init__()
        self.n_banks = config.n_banks
        self.rank = config.rank
        self.d_in = d_in
        self.d_out = d_out
        self.enable_plasticity = config.enable_plasticity
        
        # Gating network: x -> logits -> softmax -> alpha
        self.gate = nn.Linear(d_in, config.n_banks)
        
        # Weight Banks
        if self.enable_plasticity:
            # Bank 0 is plastic (buffer), Banks 1..K-1 are static (param)
            self.n_static = config.n_banks - 1
            self.plastic_bank = PlasticLowRankBank(d_in, d_out, config.rank, lr=config.plastic_lr)
            # Static banks
            self.U = nn.Parameter(torch.randn(self.n_static, d_in, config.rank) * 0.02)
            self.V = nn.Parameter(torch.randn(self.n_static, config.rank, d_out) * 0.02)
        else:
            # All static
            self.U = nn.Parameter(torch.randn(config.n_banks, d_in, config.rank) * 0.02)
            self.V = nn.Parameter(torch.randn(config.n_banks, config.rank, d_out) * 0.02)
            
        self.bias = nn.Parameter(torch.zeros(d_out))
        
    def forward(self, x):
        # x: [B, S, d_in]
        
        # 1. Compute Gating
        gate_logits = self.gate(x)
        alpha = F.softmax(gate_logits, dim=-1) # [B, S, K]
        
        # 2. Compute Expert Outputs
        if self.enable_plasticity:
            # Plastic Path (Bank 0)
            plastic_out = self.plastic_bank(x) # [B, S, d_out]
            plastic_out = plastic_out.unsqueeze(2) # [B, S, 1, d_out]
            
            # Static Path (Banks 1..K-1)
            inter = torch.einsum('bsd,kdr->bskr', x, self.U)
            static_out = torch.einsum('bskr,kro->bsko', inter, self.V) # [B, S, K-1, d_out]
            
            # Combine
            out_k = torch.cat([plastic_out, static_out], dim=2) # [B, S, K, d_out]
            
            # Online Update Trigger (Hebbian)
            if self.training:
                # Use gating activation as proxy for "relevance"
                # If gate[0] is high, this bank contributed -> update it
                # Using simple local rule: x_pre = x, y_proxy = gate[0] * output
                # We need a gradient proxy.
                # Simplification: Update plastic bank if it was active.
                # We update using input x and the layer's *final output* (feedback) or *its own output* (reinforcement).
                # Hebbian: fire together wire together.
                # Update with x and (plastic_out * alpha[0]).
                
                # We detach to avoid graph cycles if this were differentiable plasticity (it's manual here)
                relevance = alpha[:, :, 0:1] # [B, S, 1]
                weighted_out = plastic_out.squeeze(2) * relevance
                self.plastic_bank.update(x, weighted_out)
                
        else:
            # All static
            inter = torch.einsum('bsd,kdr->bskr', x, self.U)
            out_k = torch.einsum('bskr,kro->bsko', inter, self.V)
        
        # 3. Weighted Sum
        y = torch.einsum('bsko,bsk->bso', out_k, alpha)
        
        return y + self.bias

class WZMAAttention(nn.Module):
    def __init__(self, config: WZMAConfig):
        super().__init__()
        self.d_head = config.d_model // config.n_heads
        self.n_heads = config.n_heads
        self.scale = self.d_head ** -0.5
        
        # Use WZMALinear for projections
        self.q_proj = WZMALinear(config.d_model, config.d_model, config)
        self.k_proj = WZMALinear(config.d_model, config.d_model, config)
        self.v_proj = WZMALinear(config.d_model, config.d_model, config)
        self.o_proj = nn.Linear(config.d_model, config.d_model)
        
    def forward(self, x, mask=None):
        B, S, D = x.shape
        Q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask is [B, 1, 1, S]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, S, D)
        return self.o_proj(out)

class WZMAEncoderLayer(nn.Module):
    def __init__(self, config: WZMAConfig):
        super().__init__()
        self.attn = WZMAAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # FFN with WZMA
        self.ff1 = WZMALinear(config.d_model, config.d_ff, config)
        self.ff2 = nn.Linear(config.d_ff, config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff2(self.act(self.ff1(self.ln2(x)))))
        return x

class WZMAEncoder(nn.Module):
    def __init__(self, config: WZMAConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([WZMAEncoderLayer(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        
    def forward(self, input_ids, attention_mask=None):
        # Gated vocab assertion for performance
        if os.getenv("WZMA_ASSERT_VOCAB", "0") == "1":
            mx = int(input_ids.max())
            if mx >= self.config.vocab_size:
                raise RuntimeError(f"Token ID {mx} >= vocab_size {self.config.vocab_size}. Tokenizer/Config mismatch.")
        
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        pos = torch.clamp(pos, 0, self.config.max_seq_len - 1)
        
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        
        if attention_mask is not None:
            ext_mask = attention_mask[:, None, None, :]
        else:
            ext_mask = None
            
        for layer in self.layers:
            x = layer(x, ext_mask)
            
        x = self.ln_f(x)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            embeddings = x.mean(dim=1)
            
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings