"""
W-bit Native Memory Benchmark
Comparing 3 Architectures for On-Chip Context Storage:
1. ReRAM (Matrix)
2. Holographic (Superposition)
3. SDM (Sparse Distributed)
"""

import torch
import torch.nn.functional as F
import math
# import matplotlib.pyplot as plt # Optional

class MemoryBackend:
    def reset(self): pass
    def write(self, k, v, pos): pass
    def read(self, q, pos): pass
    def name(self): return "Base"

# 1. ReRAM (Baseline Matrix)
# Stores every token. Perfect recall until OOM.
class ReRAM_KV(MemoryBackend):
    def __init__(self, dim):
        self.dim = dim
        self.keys = []
        self.values = []
        
    def reset(self):
        self.keys = []
        self.values = []
        
    def name(self): return "ReRAM (Matrix)"
        
    def write(self, k, v, pos):
        self.keys.append(k)
        self.values.append(v)
        
    def read(self, q, pos):
        # Attention: Softmax(q @ K.T) @ V
        if not self.keys: return torch.zeros_like(q)
        
        K = torch.stack(self.keys)
        V = torch.stack(self.values)
        
        # Scaling
        scale = 1.0 / math.sqrt(self.dim)
        attn = F.softmax((q @ K.T) * scale, dim=-1)
        
        out = attn @ V
        return out

# 2. Holographic (HRR)
# Compresses sequence into fixed-size trace using binding.
# Trace = Sum(Key * Pos)
# Retrieval = Trace * Inverse(Pos) ~ Key
# "Binding" here is Element-wise Multiply (simplification of Circular Conv) for speed
# or full Circular Conv. Let's use Element-wise with Complex vectors (Plate/HRR) for real power
# OR simple Orthogonal Random Matrices for binding.
# Let's use Random Orthogonal Matrix binding: Binding(v, p) = v @ R_p
class Holographic_KV(MemoryBackend):
    def __init__(self, dim):
        self.dim = dim
        self.trace = torch.zeros(dim)
        # Position codebook
        self.max_len = 10000
        self.pos_emb = torch.randn(self.max_len, dim)
        self.pos_emb = F.normalize(self.pos_emb, p=2, dim=-1)
        
    def reset(self):
        self.trace = torch.zeros(self.dim)
        
    def name(self): return "Holographic (Superposition)"
    
    def write(self, k, v, pos):
        # Bind Key to Position: k * p (element-wise for simplicity in this demo, usually requires R matrix)
        # Real HRR uses Circular Convolution.
        # Let's emulate "Binding" by simple Hadamard (element-wise) with randomized position vector.
        # Note: If p is +1/-1, this is robust.
        
        p = self.pos_emb[pos % self.max_len]
        # Bind: k (*) p
        bound = k * p 
        # Add to trace (Superposition)
        self.trace += bound
        
        # We don't store V separately in pure HRR? 
        # For KV cache, we usually associate K->V. 
        # Holographic KV: Trace stores pairs (K_i * V_i).
        # Query Q matches K_i.
        # This is harder.
        # Alternative: We are just testing "Context Storage". Can we recall 'v' given 'k'?
        # Let's stick to the "Sequence Recall" task: Retrieve item at Pos P.
        
    def read(self, q, pos):
        # Unbind: Trace * Inverse(P)
        # With +/-1 vectors, Inverse(P) = P.
        # With real vectors, approx inverse is P (if orthogonal).
        p = self.pos_emb[pos % self.max_len]
        retrieved = self.trace * p
        return retrieved

# 3. SDM (Sparse Distributed Memory - Kanerva)
# Fixed set of "Hard Locations". Input addresses activate subset.
class SDM_KV(MemoryBackend):
    def __init__(self, dim, num_locs=1000):
        self.dim = dim
        self.num_locs = num_locs
        # Random addresses
        self.addresses = torch.randn(num_locs, dim)
        self.addresses = (self.addresses > 0).float() * 2 - 1 # Binary bipolar +1/-1
        # Counters
        self.counters = torch.zeros(num_locs, dim)
        
    def reset(self):
        self.counters.zero_()
        
    def name(self): return f"SDM ({self.num_locs} Locs)"
        
    def _activate(self, addr):
        # Hamming distance (dot product for bipolar)
        # addr: [dim]
        # sim: [num_locs]
        sim = self.addresses @ addr
        # Top-K activation (e.g. top 10%)
        k = max(1, int(self.num_locs * 0.05))
        _, indices = torch.topk(sim, k)
        return indices
        
    def write(self, k, v, pos):
        # Address is the Key 'k' (bipolarized)
        addr = (k > 0).float() * 2 - 1
        indices = self._activate(addr)
        
        # Write 'v' to activated counters
        # v: [dim]
        # counters[idx] += v
        self.counters[indices] += v.unsqueeze(0)
        
    def read(self, q, pos):
        # Address is Query 'q'
        addr = (q > 0).float() * 2 - 1
        indices = self._activate(addr)
        
        # Read sum from counters
        readout = self.counters[indices].sum(dim=0)
        return readout

def run_benchmark():
    DIM = 256 # W-bit Tile Width
    SEQ_LENS = [10, 50, 100, 200, 500]
    
    backends = [
        ReRAM_KV(DIM),
        Holographic_KV(DIM),
        SDM_KV(DIM, num_locs=2048) # 2048 rows (8 tiles worth)
    ]
    
    print(f"{'Backend':<25} | {'SeqLen':<5} | {'Recall':<6} | {'Mem(KB)':<8}")
    print("-" * 55)
    
    for backend in backends:
        for n in SEQ_LENS:
            backend.reset()
            
            # Generate Sequence
            # Key = Position-based (easy) or Random? Random is harder.
            # Value = Target
            keys = torch.randn(n, DIM)
            values = torch.randn(n, DIM)
            
            # Write
            for i in range(n):
                backend.write(keys[i], values[i], i)
                
            # Read (Recall All)
            hits = 0
            for i in range(n):
                # For ReRAM: query is key
                # For Holographic: query is meaningless if positional, or key if associative.
                # Let's standardize: 
                # Task: Associative Memory. Given K, retrieve V.
                
                if isinstance(backend, Holographic_KV):
                    # Holographic here implemented as Positional Storage. 
                    # To fetch value at pos i...
                    # Let's change task to Positional Recall for Holo
                    retrieved = backend.read(None, i) 
                    target = keys[i] # In Holo implementation above, we stored K at P.
                else:
                    retrieved = backend.read(keys[i], i)
                    target = values[i]
                
                # Cosine Similarity check
                sim = F.cosine_similarity(retrieved.unsqueeze(0), target.unsqueeze(0))
                if sim > 0.9: # Threshold for "Recall"
                    hits += 1
            
            acc = hits / n
            
            # Est Memory
            if isinstance(backend, ReRAM_KV):
                mem_kb = (n * DIM * 4 * 2) / 1024 # K+V, float32
            elif isinstance(backend, Holographic_KV):
                mem_kb = (DIM * 4) / 1024 # Trace only
            elif isinstance(backend, SDM_KV):
                mem_kb = (2048 * DIM * 4) / 1024 # Counters
                
            print(f"{backend.name():<25} | {n:<5} | {acc:.2f}   | {mem_kb:<8.1f}")

if __name__ == "__main__":
    run_benchmark()
