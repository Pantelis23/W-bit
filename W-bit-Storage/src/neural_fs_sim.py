"""
Neural File System (NFS) Simulator v2
Implementing the "Vector-Native" OS paradigm.

New Features:
1. Zero-Copy Querying (Semantic Search vs Byte Read).
2. "Dreaming" (Index Consolidation/Sleep Optimization).
3. Semantic Security (Data Enclaves).
"""

import os
import sys
import torch
import torch.nn.functional as F
import hashlib
import json
import random

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'W-bit', 'src'))
# Also add local src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wzma_reference.model import WZMAEncoder, WZMAConfig
from wzma_reference.tokenizer import WZMATokenizer

class NeuralBlock:
    def __init__(self, block_id, data, embedding, semantic_only=False):
        self.id = block_id
        self.size = len(data)
        self.data = data
        self.embedding = embedding
        self.semantic_only = semantic_only # If True, raw read is denied
        self.ref_count = 1

class NeuralFileSystem:
    def __init__(self):
        # 1. The "ReRAM" Index (Fast Vector Search)
        self.index_keys = []   # List of embeddings [N, Dim]
        self.index_ids = []    # List of block IDs
        
        # 2. The "NAND" Bulk Storage
        self.blocks = {}       # id -> NeuralBlock
        
        # 3. The "File Table" (Inode)
        self.files = {}        # filename -> [block_id, ...]
        
        # 4. The Controller (W-bit Chip)
        self.config = WZMAConfig(d_model=384, n_layers=2)
        self.model = WZMAEncoder(self.config)
        self.model.eval()
        self.tokenizer = self._mock_tokenizer
        
        self.total_bytes_written = 0
        
    def _mock_tokenizer(self, text):
        # Hash text to ints for stability
        ids = [ord(c) % 1000 for c in text[:128]]
        if not ids: ids = [0]
        return torch.tensor([ids], dtype=torch.long), None

    def _embed(self, text):
        with torch.no_grad():
            input_ids, _ = self.tokenizer(text)
            emb = self.model(input_ids)
            return F.normalize(emb, p=2, dim=1)

    def write_file(self, filename, content, semantic_only=False):
        self.total_bytes_written += len(content)
        data = content.encode('utf-8')
        emb = self._embed(content)
        
        # Semantic Deduplication
        is_dup = False
        dup_id = None
        
        if len(self.index_keys) > 0:
            keys = torch.cat(self.index_keys)
            sims = torch.matmul(keys, emb.T).squeeze(1)
            val, idx = torch.max(sims, dim=0)
            if val > 0.999: # Exact meaning match
                is_dup = True
                dup_id = self.index_ids[idx]
                
        if is_dup:
            print(f"[NFS] Deduped '{filename}' -> Block {dup_id}")
            self.blocks[dup_id].ref_count += 1
            self.files[filename] = dup_id
        else:
            new_id = len(self.blocks)
            access = "SECURE" if semantic_only else "OPEN"
            print(f"[NFS] Writing '{filename}' [{access}] -> New Block {new_id}")
            
            block = NeuralBlock(new_id, data, emb, semantic_only)
            self.blocks[new_id] = block
            self.index_keys.append(emb)
            self.index_ids.append(new_id)
            self.files[filename] = new_id

    def read_file(self, filename):
        """Standard OS Read. Fails if block is Semantic Only."""
        bid = self.files.get(filename)
        if bid is None: return None
        
        blk = self.blocks[bid]
        if blk.semantic_only:
            raise PermissionError(f"ACCESS_DENIED: '{filename}' is a Semantic Enclave. Use query().")
        
        return blk.data.decode('utf-8')

    def query(self, text):
        """Zero-Copy Semantic Search. Works on all files."""
        q_emb = self._embed(text)
        
        if not self.index_keys: return None
        
        # Vector Search in ReRAM (Instant)
        keys = torch.cat(self.index_keys)
        sims = torch.matmul(keys, q_emb.T).squeeze(1)
        val, idx = torch.max(sims, dim=0)
        
        bid = self.index_ids[idx]
        blk = self.blocks[bid]
        
        print(f"[NFS] Query '{text}' -> Matched Block {bid} (Sim: {val:.4f})")
        # In a real W-bit, this returns a generative answer.
        # Here we return the raw content snippet to prove access.
        return blk.data.decode('utf-8')

    def dream(self):
        """
        Sleep Mode Optimization.
        Consolidates index vectors that are very close (drift correction).
        """
        print("\n[NFS] Entering Dream State (Consolidation)...")
        before_count = len(self.index_keys)
        
        # Simple clustering: If key A and B are close, keep A, point B's block to A's key?
        # No, index_keys maps 1:1 to index_ids.
        # We want to remove redundant keys from the INDEX (ReRAM space is precious).
        # If Block 1 and Block 2 have similar embeddings (but different data), 
        # we can't merge them.
        # BUT if we have many small fragments that form a topic, we can create a "Topic Centroid".
        
        # For simulation: Remove vectors that are too close to others (Sim > 0.95)
        # This simulates "Pruning" irrelevant/redundant memories.
        
        if not self.index_keys: return
        
        kept_keys = []
        kept_ids = []
        
        keys = torch.cat(self.index_keys)
        n = keys.shape[0]
        mask = torch.ones(n, dtype=torch.bool)
        
        for i in range(n):
            if not mask[i]: continue
            
            # Check similarity with remaining
            current = keys[i:i+1]
            sims = torch.matmul(keys, current.T).squeeze(1)
            
            # Mark close neighbors as "merged"
            neighbors = (sims > 0.95) & mask
            neighbors[i] = False # Don't remove self
            
            if neighbors.any():
                removed = torch.where(neighbors)[0]
                # print(f"  Merging indices {removed.tolist()} into {i}")
                mask[neighbors] = False
                
            kept_keys.append(current)
            kept_ids.append(self.index_ids[i])
            
        self.index_keys = kept_keys
        self.index_ids = kept_ids
        
        print(f"  Index reduced: {before_count} -> {len(self.index_keys)} entries.")

def run_demo():
    fs = NeuralFileSystem()
    
    # 1. Secure Storage
    secret = "The nuclear launch code is 12345."
    fs.write_file("secrets.txt", secret, semantic_only=True)
    
    # 2. Standard Storage
    public = "The cafeteria menu is pizza."
    fs.write_file("menu.txt", public, semantic_only=False)
    
    # 3. Security Check
    print("\n--- Security Check ---")
    try:
        print(f"Reading menu: {fs.read_file('menu.txt')}")
        print("Reading secrets...", end=" ")
        fs.read_file("secrets.txt")
    except PermissionError as e:
        print(e)
        
    # 4. Semantic Bypass (The "Legal" way)
    # The drive answers questions without revealing raw bytes
    print("\n--- Semantic Query ---")
    ans = fs.query("What is the code?")
    print(f"Drive Answered: {ans}")
    
    # 5. Dreaming
    # Add redundant info
    fs.write_file("menu_v2.txt", "The cafeteria menu is pizza.", semantic_only=False)
    fs.write_file("menu_v3.txt", "The cafeteria menu is pizza.", semantic_only=False)
    
    fs.dream()

if __name__ == "__main__":
    run_demo()
