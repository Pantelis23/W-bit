"""
Mixed Dataset for WZMA Training
Combines:
1. Invariance Task: (Code, AugCode)
2. Alignment Task: (Docstring, Code)
"""
import torch
import random
import os
import ast
from torch.utils.data import Dataset
from src.agi.memory.wzma_embedder.data import SyntheticTextDataset

# Reuse the robust extraction logic
def get_code_body_clean(source, node):
    try:
        if not getattr(node, "end_lineno", None): return None
        lines = source.splitlines(True)
        start = node.lineno - 1
        end = node.end_lineno
        if start >= len(lines) or end > len(lines): return None
        seg = lines[start:end]
        
        # remove docstring stmt
        if node.body and isinstance(node.body[0], ast.Expr):
             # Check if it's a docstring constant
             val = getattr(node.body[0], "value", None)
             if isinstance(val, ast.Constant) and isinstance(val.value, str):
                ds_start = node.body[0].lineno - 1
                ds_end = node.body[0].end_lineno
                rel_start = ds_start - start
                rel_end = ds_end - start
                if rel_start >= 0 and rel_end <= len(seg):
                    for i in range(rel_start, rel_end): seg[i] = ""
        
        return "".join(seg).strip()
    except:
        return None

class MixedCodeDataset(Dataset):
    def __init__(self, size=1000, tokenizer=None, max_len=384, root_dirs=["src"], ratio_align=0.5):
        self.size = size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ratio_align = ratio_align
        
        # 1. Base synthetic dataset for invariance (streamed corpus)
        # We need a corpus path. Let's find one.
        corpus_path = "checkpoints/wzma_embedder/corpus.txt"
        self.invariance_ds = SyntheticTextDataset(size=size, tokenizer=None, corpus_path=corpus_path)
        
        # 2. Alignment pairs (in-memory, scanned once)
        self.align_pairs = []
        self._scan_pairs(root_dirs)
        print(f"Scanned {len(self.align_pairs)} alignment pairs (Doc->Code).")
        
    def _scan_pairs(self, root_dirs):
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                # Try relative
                if os.path.exists(os.path.join("..", root_dir)):
                    root_dir = os.path.join("..", root_dir)
                    
            for root, dirs, files in os.walk(root_dir):
                if '.venv' in root or '__pycache__' in root: continue
                for file in files:
                    if file.endswith(".py"):
                        path = os.path.join(root, file)
                        try:
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                source = f.read()
                                tree = ast.parse(source)
                            for node in ast.walk(tree):
                                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                    doc = ast.get_docstring(node)
                                    if doc and len(doc) > 20:
                                        code = get_code_body_clean(source, node)
                                        if code and len(code) > 20:
                                            # Truncate
                                            self.align_pairs.append((doc[:500], code[:1000]))
                        except:
                            continue

    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Decide task type
        is_align = (random.random() < self.ratio_align) and (len(self.align_pairs) > 0)
        
        if is_align:
            # Task: Alignment
            doc, code = random.choice(self.align_pairs)
            anchor_text = f"DOC: {doc}"
            positive_text = f"CODE: {code}"
        else:
            # Task: Invariance
            # Helper generates random line from corpus
            anchor_raw = self.invariance_ds._generate_sentence()
            positive_raw = self.invariance_ds._augment(anchor_raw)
            anchor_text = f"CODE: {anchor_raw}"
            positive_text = f"CODE: {positive_raw}"
            
        # Tokenize
        if self.tokenizer:
            a_ids, a_mask = self.tokenizer.encode(anchor_text, self.max_len)
            p_ids, p_mask = self.tokenizer.encode(positive_text, self.max_len)
            return {
                "anchor_ids": torch.tensor(a_ids, dtype=torch.long),
                "anchor_mask": torch.tensor(a_mask, dtype=torch.long),
                "positive_ids": torch.tensor(p_ids, dtype=torch.long),
                "positive_mask": torch.tensor(p_mask, dtype=torch.long)
            }
        else:
            return anchor_text, positive_text