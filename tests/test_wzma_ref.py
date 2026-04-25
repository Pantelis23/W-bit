import os
import torch
import pytest
import sys
import shutil

# Add W-bit root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.wzma_reference.tokenizer import WZMATokenizer
from src.wzma_reference.model import WZMAEncoder, WZMAConfig

@pytest.fixture
def tokenizer():
    tok = WZMATokenizer(vocab_size=1000)
    with open("tmp_corpus.txt", "w") as f:
        f.write("hello world\n" * 10)
    tok.train(["tmp_corpus.txt"])
    yield tok
    if os.path.exists("tmp_corpus.txt"): os.remove("tmp_corpus.txt")

def test_tokenizer_idempotence(tokenizer):
    save_path = "tmp_tokenizer.json"
    tokenizer.save(save_path)
    tok2 = WZMATokenizer(tokenizer_path=save_path)
    assert tok2.vocab_size == tokenizer.vocab_size
    ids1, _ = tokenizer.encode("hello complexity")
    ids2, _ = tok2.encode("hello complexity")
    assert ids1 == ids2
    os.remove(save_path)

def test_model_vocab_assertion():
    config = WZMAConfig(vocab_size=100)
    model = WZMAEncoder(config)
    
    # Enable assertion
    os.environ["WZMA_ASSERT_VOCAB"] = "1"
    
    # 1. Normal forward
    ids = torch.tensor([[10, 20]], dtype=torch.long)
    model(ids) 
    
    # 2. OOB forward
    bad_ids = torch.tensor([[10, 150]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="Token ID"):
        model(bad_ids)
        
    os.environ.pop("WZMA_ASSERT_VOCAB", None)

def test_load_production_checkpoint():
    """Verify we can load the production checkpoint transferred from AtlasLM."""
    ckpt_path = "checkpoints/wzma_reference/model.pt"
    tok_path = "checkpoints/wzma_reference/tokenizer.json"
    
    if not os.path.exists(ckpt_path):
        pytest.skip("Production checkpoint not found")
        
    # 1. Load Tokenizer
    tokenizer = WZMATokenizer(tokenizer_path=tok_path)
    
    # 2. Load Checkpoint
    # We use CPU mapping for test stability
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    # 3. Verify Handshake
    assert "tokenizer_fingerprint" in ckpt
    assert "config" in ckpt
    
    expected_fp = ckpt["tokenizer_fingerprint"]
    actual_fp = tokenizer.get_fingerprint()
    assert actual_fp == expected_fp, "Fingerprint mismatch in transferred artifacts!"
    
    # 4. Init Model
    config = WZMAConfig(**ckpt["config"])
    model = WZMAEncoder(config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    # 5. Smoke Inference
    ids, mask = tokenizer.encode("W-bit Hardware Reference")
    ids_t = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(ids_t)
    
    assert out.shape == (1, 384) # Dim check
    print("Production checkpoint verified.")

if __name__ == "__main__":
    pass