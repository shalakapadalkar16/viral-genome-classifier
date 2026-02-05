"""
PyTorch Dataset for genomic sequences
"""
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from pathlib import Path
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ViralGenomeDataset(Dataset):
    """Dataset for viral genome sequences"""
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        tokenizer,
        max_length: int = 1024,
        label_to_id: Dict[str, int] = None
    ):
        """
        Initialize dataset
        
        Args:
            metadata_df: DataFrame with sequence metadata
            tokenizer: Tokenizer for sequences (ESM or DNABERT)
            max_length: Maximum sequence length
            label_to_id: Mapping from label names to IDs
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        if label_to_id is None:
            unique_labels = sorted(self.metadata_df['label'].unique())
            self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_id = label_to_id
        
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # Load sequences into memory (assuming dataset fits)
        self.sequences = self._load_sequences()
        
        logger.info(f"Loaded {len(self.sequences)} sequences")
        logger.info(f"Label mapping: {self.label_to_id}")
    
    def _load_sequences(self) -> List[str]:
        """Load all sequences from FASTA files"""
        sequences = []
        
        # Group by file
        for file_path in self.metadata_df['file'].unique():
            seq_ids = set(self.metadata_df[self.metadata_df['file'] == file_path]['id'])
            
            # Load sequences from FASTA
            for record in SeqIO.parse(file_path, "fasta"):
                if record.id in seq_ids:
                    sequences.append(str(record.seq))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        # Get sequence and label
        sequence = self.sequences[idx]
        label_name = self.metadata_df.iloc[idx]['label']
        label_id = self.label_to_id[label_name]
        
        # Tokenize sequence
        # Truncate if necessary
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        # Tokenize (handles padding automatically)
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_id, dtype=torch.long),
            'sequence_id': self.metadata_df.iloc[idx]['id']
        }