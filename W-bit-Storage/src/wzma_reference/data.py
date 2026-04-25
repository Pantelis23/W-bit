"""
Data Generation for WZMA Embedder
Generates synthetic positive pairs for contrastive learning.
Supports synthetic text generation AND loading from corpus via stable O(1) random binary seek.
"""
import random
import torch
import os
from torch.utils.data import Dataset, DataLoader

class SyntheticTextDataset(Dataset):
    def __init__(self, size=1000, tokenizer=None, max_len=128, corpus_path=None):
        self.size = size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.corpus_path = corpus_path
        self.vocab = ["AI", "neural", "network", "learning", "memory", "gpu", "cpu", "tensor", "matrix", "vector", "embedding", "search", "retrieval", "agent", "robot", "human", "code", "python", "rust", "optimization", "gradient", "loss", "accuracy", "training", "inference", "model", "layer", "transformer", "attention", "wzma", "aeternum", "morphium", "atlas", "evoluma"]
        
        self.corpus_size = 0
        self._fh = None
        
        if self.corpus_path and os.path.exists(self.corpus_path):
            self.corpus_size = os.path.getsize(self.corpus_path)
            
    def _get_fh(self):
        # Open file handle lazily per worker, binary mode for stable seeking
        if self._fh is None and self.corpus_path:
            self._fh = open(self.corpus_path, "rb", buffering=1024*1024)
        return self._fh

    def _generate_sentence(self):
        if self.corpus_size > 0:
            f = self._get_fh()
            max_off = max(0, self.corpus_size - 1)
            
            # Try a few times to find a valid line
            for _ in range(8):
                off = random.randint(0, max_off)
                f.seek(off)
                f.readline() # Discard partial
                line = f.readline() # Read full line
                
                if not line: continue
                
                try:
                    s = line.decode("utf-8", errors="ignore").strip()
                    if len(s) > 10:
                        return s
                except:
                    continue
        
        # Fallback to synthetic vocab
        length = random.randint(5, 15)
        return " ".join(random.choices(self.vocab, k=length))
        
    def _augment(self, text):
        words = text.split()
        if len(words) < 2: return text
        
        aug_type = random.choice(['dropout', 'span_dropout', 'shuffle', 'noise'])
        
        if aug_type == 'dropout':
            # Drop 10-30% of words
            mask = [random.random() > 0.2 for _ in words]
            new_words = [w for w, m in zip(words, mask) if m]
            if not new_words: new_words = words
            return " ".join(new_words)
            
        elif aug_type == 'span_dropout':
            # Drop a contiguous span
            if len(words) > 3:
                start = random.randint(0, len(words)-2)
                end = min(len(words), start + random.randint(1, 3))
                new_words = words[:start] + words[end:]
                if not new_words: new_words = words
                return " ".join(new_words)
            return " ".join(words)
            
        elif aug_type == 'shuffle':
            # Only if sentence is long enough to preserve semantics somewhat
            if len(words) > 5:
                # Local shuffle window? Or full? Full shuffle kills semantics.
                # Let's shuffle middle 50%
                mid_s = len(words) // 4
                mid_e = len(words) - mid_s
                mid = words[mid_s:mid_e]
                random.shuffle(mid)
                new_words = words[:mid_s] + mid + words[mid_e:]
                return " ".join(new_words)
            return " ".join(words)
            
        else: # Noise
            # Inject punctuation or typos
            idx = random.randint(0, len(words)-1)
            words[idx] = words[idx] + random.choice(['.', ',', '!', '?'])
            return " ".join(words)

    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Anchor
        anchor_text = self._generate_sentence()
        # Positive (augmented)
        positive_text = self._augment(anchor_text)
        
        if self.tokenizer:
            # On-the-fly tokenization
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

def create_corpus_file(path, lines=5000):
    # Dummy synthetic
    ds = SyntheticTextDataset(size=lines)
    with open(path, "w") as f:
        for _ in range(lines):
            f.write(ds._generate_sentence() + "\n")