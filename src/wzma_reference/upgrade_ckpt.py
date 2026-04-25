"""
Upgrade WZMA Checkpoint
Adds tokenizer fingerprint and config metadata to legacy checkpoints.
"""
import sys
import os
import torch
import json
import argparse
import tokenizers
import platform
import time
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agi.memory.wzma_embedder.tokenizer import (
    WZMATokenizer,
    CURRENT_FINGERPRINT_VERSION,
)
from src.agi.memory.wzma_embedder.model import WZMAConfig

def upgrade_checkpoint(args):
    print(f"Upgrading checkpoint: {args.ckpt_path}")
    
    if not os.path.exists(args.ckpt_path):
        print("Checkpoint not found.")
        sys.exit(1)
        
    # Load components
    tokenizer = WZMATokenizer(tokenizer_path=args.tokenizer_path)
    print(f"Loaded tokenizer. Fingerprint: {tokenizer.get_fingerprint()[:8]} (v{CURRENT_FINGERPRINT_VERSION})")
    
    # Load state dict
    ckpt = torch.load(
        args.ckpt_path, 
        map_location="cpu", 
        weights_only=(not args.unsafe_load)
    )
    
    # 1. Extract state dict and potential config
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        ckpt_config = ckpt.get("config")
        print("Checkpoint already has metadata structure.")
    else:
        state_dict = ckpt
        ckpt_config = None
        print("Legacy checkpoint detected.")

    # 2. Determine Config
    config_dict = None
    if ckpt_config:
        config_dict = ckpt_config
    elif args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config_dict = json.load(f)
    
    if config_dict is None:
        print("WARN: No config found. Inferring from tokenizer...")
        config_dict = {"vocab_size": tokenizer.get_real_vocab_size()}
    
    config = WZMAConfig(**config_dict)
    
    # Create new payload
    payload = {
        "state_dict": state_dict,
        "tokenizer_fingerprint": tokenizer.get_fingerprint(),
        "fingerprint_version": CURRENT_FINGERPRINT_VERSION,
        "config": config.__dict__,
        "meta": {
            "tokenizers": str(tokenizers.__version__),
            "python": str(platform.python_version()),
            "torch": str(torch.__version__),
        }
    }
    
    # 3. Save atomically
    out_path = args.out_path or args.ckpt_path
    
    # Timestamped backup if overwriting existing file
    if out_path == args.ckpt_path and os.path.exists(out_path):
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = f"{out_path}.bak.{ts}"
        try:
            shutil.copy2(out_path, bak)
            print(f"Backup created at {bak}")
        except Exception as e:
            print(f"WARN: Failed to create backup: {e}")

    tmp_path = out_path + ".tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
        print(f"Success. Saved to {out_path}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/wzma_embedder/tokenizer.json")
    parser.add_argument("--config_path", type=str, default="checkpoints/wzma_embedder/config.json")
    parser.add_argument("--out_path", type=str, help="Output path (defaults to overwrite)")
    parser.add_argument("--unsafe_load", action="store_true", help="Use weights_only=False.")
    args = parser.parse_args()
    upgrade_checkpoint(args)