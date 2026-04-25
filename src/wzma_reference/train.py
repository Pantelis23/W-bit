"""
Training script for WZMA Embedder
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
from .data import SyntheticTextDataset, create_corpus_file
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
    print(f"--- Training WZMA Embedder on {args.device} ---")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Tokenizer
    tok_out_path = os.path.join(args.out_dir, "tokenizer.json")
    corpus_path = args.corpus_path
    if not corpus_path:
        corpus_path = os.path.join(args.out_dir, "corpus.txt")
        if not os.path.exists(corpus_path):
            create_corpus_file(corpus_path, lines=5000)
    
    if args.tokenizer_path:
        tokenizer = WZMATokenizer(tokenizer_path=args.tokenizer_path)
        print(f"Loaded tokenizer from {args.tokenizer_path} (Vocab: {tokenizer.vocab_size})")
        tokenizer.save(tok_out_path)
    else:
        tokenizer = WZMATokenizer()
        if os.path.exists(tok_out_path) and not args.force_retrain_tokenizer and not args.train_tokenizer:
            print("Loading existing tokenizer from out_dir...")
            tokenizer.load(tok_out_path)
        else:
            print(f"Training tokenizer from scratch on {corpus_path}...")
            tokenizer.train([corpus_path])
            tokenizer.save(tok_out_path)
            print(f"Trained tokenizer. Actual Vocab Size: {tokenizer.vocab_size}")
        
    # 2. Model
    actual_vocab_size = tokenizer.get_real_vocab_size()
    
    config = WZMAConfig(
        vocab_size=actual_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads, 
        n_banks=args.n_banks,
        rank=args.rank
    )
    
    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)
        
    model = WZMAEncoder(config).to(args.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. Data
    dataset = SyntheticTextDataset(size=args.steps * args.batch_size, tokenizer=tokenizer, corpus_path=corpus_path)
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
            
        if step % 1000 == 0 and step > 0:
            save_checkpoint(model, tokenizer, config, os.path.join(args.out_dir, f"ckpt_{step}.pt"))
            
        step += 1
        
    # Save final
    save_checkpoint(model, tokenizer, config, os.path.join(args.out_dir, "model.pt"))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="checkpoints/wzma_embedder")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_path", type=str, help="Path to existing tokenizer")
    parser.add_argument("--train_tokenizer", action="store_true", help="Force train new tokenizer from scratch")
    parser.add_argument("--force_retrain_tokenizer", action="store_true", help="Legacy flag")
    parser.add_argument("--corpus_path", type=str, help="Path to text corpus for training tokenizer")
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_banks", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    train(args)
