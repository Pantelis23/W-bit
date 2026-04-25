"""
Tokenizer for WZMA Embedder
Wraps a HuggingFace Tokenizer (BPE/ByteLevel) for WZMA.
Supports loading existing tokenizers or training from scratch.
"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os
import json
import hashlib

CURRENT_FINGERPRINT_VERSION = 1

class WZMATokenizer:
    def __init__(self, vocab_size=8192, tokenizer_path=None):
        self.vocab_size = vocab_size
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}...")
            self.load(tokenizer_path, allow_mutation=False)
        else:
            # Initialize empty for training
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    def get_fingerprint(self) -> str:
        """
        Stable hash of full tokenizer state (model+merges+normalizer+pretokenizer+postproc+added_tokens).
        Canonicalized JSON to reduce whitespace/key-order drift.
        """
        try:
            raw = self.tokenizer.to_str()
            # Canonicalize JSON
            obj = json.loads(raw)
            s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            # Fallback
            vocab = sorted(self.tokenizer.get_vocab().items())
            s = json.dumps(vocab, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _sanity_check_vocab(self, rv: int):
        """Strict adversarial check for vocab coverage."""
        for t in self.special_tokens:
            tid = self.tokenizer.token_to_id(t)
            assert tid is not None, f"Strict check failed: missing special token {t}"
            assert tid < rv, f"Strict check failed: special {t} id {tid} >= {rv}"

        probes = [
            "".join(chr(i) for i in range(256)),
            "π∑∞—✓😈\u0000\u0001\u0002",
            "a" * 4096,
        ]
        for p in probes:
            enc = self.tokenizer.encode(p)
            ids = enc.ids
            if ids:
                mx = max(ids)
                assert mx < rv, f"Tokenizer produced ID {mx} >= vocab_size {rv} for probe text"

    def get_real_vocab_size(self):
        """Returns max_token_id + 1 to ensure safe embedding table size."""
        vocab = self.tokenizer.get_vocab()
        if not vocab:
            return 0
        rv = max(vocab.values()) + 1
        self._sanity_check_vocab(rv)
        return rv

    def _ensure_special_tokens(self, add_if_missing: bool = True):
        """Ensure special tokens exist. Raises error or adds them."""
        missing = []
        for token in self.special_tokens:
            if self.tokenizer.token_to_id(token) is None:
                missing.append(token)
        
        if missing:
            if add_if_missing:
                print(f"Adding missing special tokens: {missing}")
                self.tokenizer.add_special_tokens(missing)
            else:
                raise ValueError(f"Tokenizer missing special tokens: {missing}. Cannot load safely.")

    def train(self, files):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size, 
            special_tokens=self.special_tokens
        )
        self.tokenizer.train(files, trainer)
        self.vocab_size = self.get_real_vocab_size()
        
    def save(self, path):
        self.tokenizer.save(path)
        
    def load(self, path, allow_mutation: bool = False):
        """Load tokenizer. Mutates/adds tokens ONLY if allowed."""
        self.tokenizer = Tokenizer.from_file(path)
        self._ensure_special_tokens(add_if_missing=allow_mutation)
        self.vocab_size = self.get_real_vocab_size()
        
    def encode(self, text, max_length=128):
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        
        pad_id = self.pad_token_id
        
        # Pad/Truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
            mask = [1] * max_length
        else:
            pad_len = max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [pad_id] * pad_len
            
        return ids, mask
        
    @property
    def pad_token_id(self):
        tid = self.tokenizer.token_to_id("<PAD>")
        if tid is None:
            if self.tokenizer.get_vocab_size() == 0:
                return 0
            raise ValueError("<PAD> token missing from initialized tokenizer.")
        return tid