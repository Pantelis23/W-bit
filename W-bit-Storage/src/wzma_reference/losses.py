"""
Loss functions for WZMA Embedder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchors, positives):
        # anchors, positives: [B, D]
        # Normalize
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        
        # Similarity matrix: [B, B]
        # logits[i, j] = sim(anchor[i], positive[j])
        logits = torch.matmul(anchors, positives.T) / self.temperature
        
        # Targets: labels are 0, 1, 2, ... B-1 (diagonal)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
