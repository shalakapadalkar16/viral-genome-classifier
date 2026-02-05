"""
Simplified CNN with strong regularization
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class SimpleCNNClassifier(nn.Module):
    """Simple CNN with regularization for small datasets"""
    
    def __init__(
        self,
        vocab_size: int = 258,
        embedding_dim: int = 64,
        num_labels: int = 5,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Single CNN layer with batch norm
        self.conv = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Embedding
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)
        
        # Apply mask
        mask = attention_mask.unsqueeze(1).float()
        x = x * mask
        
        # Convolution
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        
        # Pool
        x = self.pool(x).squeeze(-1)
        
        # Classify
        logits = self.classifier(x)
        
        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': x
        }