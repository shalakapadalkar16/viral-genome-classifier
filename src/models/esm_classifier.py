"""
ESM-2 based classifier for viral genomes
"""
import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer, AutoModel
from typing import Dict, Optional


class ESM2Classifier(nn.Module):
    """ESM-2 model fine-tuned for genome classification"""
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        num_labels: int = 5,
        hidden_dropout_prob: float = 0.1,
        freeze_encoder: bool = False,
        hidden_dims: list = None
    ):
        """
        Initialize ESM-2 classifier
        
        Args:
            model_name: Pretrained ESM model name
            num_labels: Number of classification labels
            hidden_dropout_prob: Dropout probability
            freeze_encoder: Whether to freeze encoder weights
            hidden_dims: Hidden layer dimensions for classifier head
        """
        super().__init__()
        
        # Load pretrained ESM-2 model
        self.esm = EsmModel.from_pretrained(model_name)
        self.config = self.esm.config
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.esm.parameters():
                param.requires_grad = False
        
        # Classification head
        if hidden_dims is None:
            # Simple linear classifier
            self.classifier = nn.Sequential(
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(self.config.hidden_size, num_labels)
            )
        else:
            # Multi-layer classifier
            layers = []
            input_dim = self.config.hidden_size
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob)
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, num_labels))
            self.classifier = nn.Sequential(*layers)
        
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
        # Get ESM-2 embeddings
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(sequence_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': sequence_output
        }
    
    def unfreeze_encoder(self, num_layers: int = -1):
        """
        Unfreeze encoder layers for fine-tuning
        
        Args:
            num_layers: Number of top layers to unfreeze (-1 for all)
        """
        if num_layers == -1:
            # Unfreeze all
            for param in self.esm.parameters():
                param.requires_grad = True
        else:
            # Unfreeze top N layers
            total_layers = len(self.esm.encoder.layer)
            for i in range(total_layers - num_layers, total_layers):
                for param in self.esm.encoder.layer[i].parameters():
                    param.requires_grad = True