"""
Mixed Training script for WZMA Embedder
Trains on both Invariance and Alignment tasks.
"""
import argparse
import os
import torch
import torch.optim as optim
import tokenizers
import platform
from .model import WZMAEncoder, WZMAConfig
from .tokenizer import (
    WZMATokenizer,
    CURRENT_FINGERPRINT_VERSION,
)
from .data_mixed import MixedCodeDataset
from .losses import InfoNCELoss
from torch.utils.data import DataLoader
import json

def save_checkpoint(model, tokenizer, config, path):
    """Saves model with metadata handshake."""
    payload = {
        "state_dict": model.state_dict(),
        "tokenizer_fingerprint": tokenizer.get_fingerprint(),
        "fingerprint_version": CURRENT_FINGERPRINT_VERSION,
        "config": config.__dict__,
        "meta": {
            "tokenizers": str(tokenizers.__version__),
            "python": str(platform.python_version()),
            "torch": str(torch.__version__),
        }
    }
    torch.save(payload, path)

def train(args):
    print(f"--- Training WZMA Embedder (Mixed) on {args.device} ---")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Tokenizer
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("Tokenizer not found! Run train.py first.")
        return
        
    tokenizer = WZMATokenizer(tokenizer_path=tok_path)
    print(f"Loaded tokenizer (Vocab: {tokenizer.vocab_size})")
        
    # 2. Model
    config_path = os.path.join(args.out_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = WZMAConfig(**config_dict)
    else:
        config = WZMAConfig(
            vocab_size=tokenizer.get_real_vocab_size(),
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_banks=args.n_banks,
            rank=args.rank
        )
    
    model = WZMAEncoder(config).to(args.device)
    
    model_path = os.path.join(args.out_dir, "model.pt")
    if os.path.exists(model_path):
        print("Loading previous state...")
        ckpt = torch.load(model_path, map_location=args.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. Data
    dataset = MixedCodeDataset(
        size=args.steps * args.batch_size, 
        tokenizer=tokenizer,
        max_len=128,
        root_dirs=["src", "scripts", "../W-bit", "../Adaptive_OS"],
        ratio_align=0.5
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = InfoNCELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # 5. Loop
    model.train()
    step = 0
    
    for batch in dataloader:
        if step >= args.steps: break
        
        a_ids = batch["anchor_ids"].to(args.device)
        a_mask = batch["anchor_mask"].to(args.device)
        p_ids = batch["positive_ids"].to(args.device)
        p_mask = batch["positive_mask"].to(args.device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            emb_a = model(a_ids, a_mask)
            emb_p = model(p_ids, p_mask)
            loss = criterion(emb_a, emb_p)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 100 == 0:
            print(f"Step {step}/{args.steps} | Loss: {loss.item():.4f} | Mem: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            
        if step % 2000 == 0 and step > 0:
            save_checkpoint(model, tokenizer, config, os.path.join(args.out_dir, f"ckpt_mixed_{step}.pt"))
        step += 1
        
    save_checkpoint(model, tokenizer, config, os.path.join(args.out_dir, "model.pt"))
    print("Mixed Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="checkpoints/wzma_embedder")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)