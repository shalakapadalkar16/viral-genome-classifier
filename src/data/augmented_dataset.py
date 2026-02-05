"""
Dataset with data augmentation
"""
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from pathlib import Path
import pandas as pd
from typing import Dict, List
import logging

from src.data.augmentation import DNASequenceAugmenter

logger = logging.getLogger(__name__)


class AugmentedViralGenomeDataset(Dataset):
    """Dataset with data augmentation for viral genomes"""
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        tokenizer,
        max_length: int = 1024,
        label_to_id: Dict[str, int] = None,
        augment: bool = False,
        augmentation_factor: int = 3
    ):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmentation_factor = augmentation_factor
        
        if augment:
            self.augmenter = DNASequenceAugmenter()
        
        # Create label mapping
        if label_to_id is None:
            unique_labels = sorted(self.metadata_df['label'].unique())
            self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_id = label_to_id
        
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # Load sequences
        self.sequences = self._load_sequences()
        
        # If augmenting, expand dataset
        if self.augment:
            self._create_augmented_dataset()
        
        logger.info(f"Loaded {len(self)} samples (augmented: {self.augment})")
        logger.info(f"Label mapping: {self.label_to_id}")
    
    def _load_sequences(self) -> List[str]:
        """Load all sequences from FASTA files"""
        sequences = []
        
        for file_path in self.metadata_df['file'].unique():
            seq_ids = set(self.metadata_df[self.metadata_df['file'] == file_path]['id'])
            
            for record in SeqIO.parse(file_path, "fasta"):
                if record.id in seq_ids:
                    sequences.append(str(record.seq))
        
        return sequences
    
    def _create_augmented_dataset(self):
        """Create augmented versions of all sequences"""
        original_sequences = self.sequences.copy()
        original_labels = self.metadata_df['label'].tolist()
        
        augmented_sequences = []
        augmented_labels = []
        
        for seq, label in zip(original_sequences, original_labels):
            # Get augmented versions
            aug_seqs = self.augmenter.augment(seq, self.augmentation_factor - 1)
            
            augmented_sequences.extend(aug_seqs)
            augmented_labels.extend([label] * len(aug_seqs))
        
        self.sequences = augmented_sequences
        self.metadata_df = pd.DataFrame({
            'label': augmented_labels,
            'id': [f"aug_{i}" for i in range(len(augmented_labels))]
        })
        
        logger.info(f"Created {len(self.sequences)} augmented samples from {len(original_sequences)} original")
    
    def __len__(self) -> int:
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sequence = self.sequences[idx]
        label_name = self.metadata_df.iloc[idx]['label']
        label_id = self.label_to_id[label_name]
        
        # Tokenize
        if len(sequence) > self.max_length * 4:  # Approximate for k-mers
            sequence = sequence[:self.max_length * 4]
        
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None  # Get lists, not tensors
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label_id, dtype=torch.long),
            'sequence_id': self.metadata_df.iloc[idx]['id']
        }