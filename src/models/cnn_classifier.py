"""
CNN baseline for DNA sequence classification
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class DNACNNClassifier(nn.Module):
    """Simple CNN for DNA sequence classification"""
    
    def __init__(
        self,
        vocab_size: int = 4098,  # 4^6 + special tokens
        embedding_dim: int = 128,
        num_labels: int = 5,
        dropout: float = 0.3
    ):
        """
        Initialize CNN classifier
        
        Args:
            vocab_size: Size of k-mer vocabulary
            embedding_dim: Embedding dimension
            num_labels: Number of classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple CNN layers with different kernel sizes
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, 256, kernel_size=7, padding=3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Dictionary with loss and logits
        """
        # Embedding: [batch, seq_len, embed_dim]
        x = self.embedding(input_ids)
        
        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply mask
        mask = attention_mask.unsqueeze(1).float()
        x = x * mask
        
        # Multiple convolutions
        conv1_out = torch.relu(self.conv1(x))
        conv2_out = torch.relu(self.conv2(x))
        conv3_out = torch.relu(self.conv3(x))
        
        # Pool each
        pool1 = self.pool(conv1_out).squeeze(-1)
        pool2 = self.pool(conv2_out).squeeze(-1)
        pool3 = self.pool(conv3_out).squeeze(-1)
        
        # Concatenate
        pooled = torch.cat([pool1, pool2, pool3], dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': pooled
        }