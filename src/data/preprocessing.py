"""
Preprocess and validate genomic sequences
"""
import re
from typing import Tuple, List, Dict
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class GenomePreprocessor:
    """Clean and validate genomic sequences"""
    
    def __init__(
        self, 
        min_length: int = 1000,
        max_length: int = 30000,
        valid_nucleotides: str = "ATCG"
    ):
        """
        Initialize preprocessor
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            valid_nucleotides: Allowed nucleotide characters
        """
        self.min_length = min_length
        self.max_length = max_length
        self.valid_nucleotides = set(valid_nucleotides)
        
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean sequence by removing invalid characters
        
        Args:
            sequence: Raw DNA sequence
            
        Returns:
            Cleaned sequence
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Replace ambiguous nucleotides
        ambiguous_map = {
            'N': 'A',  # Unknown
            'R': 'A',  # A or G (purine)
            'Y': 'C',  # C or T (pyrimidine)
            'K': 'G',  # G or T (keto)
            'M': 'A',  # A or C (amino)
            'S': 'G',  # G or C (strong)
            'W': 'A',  # A or T (weak)
            'B': 'C',  # C, G, or T
            'D': 'A',  # A, G, or T
            'H': 'A',  # A, C, or T
            'V': 'A',  # A, C, or G
        }
        
        for ambig, replacement in ambiguous_map.items():
            sequence = sequence.replace(ambig, replacement)
        
        # Remove any remaining non-nucleotide characters
        sequence = ''.join([c for c in sequence if c in self.valid_nucleotides])
        
        return sequence
    
    def validate_sequence(self, sequence: str) -> Tuple[bool, str]:
        """
        Validate sequence quality
        
        Args:
            sequence: DNA sequence
            
        Returns:
            (is_valid, reason) tuple
        """
        # Check length
        if len(sequence) < self.min_length:
            return False, f"Too short: {len(sequence)} < {self.min_length}"
        
        if len(sequence) > self.max_length:
            return False, f"Too long: {len(sequence)} > {self.max_length}"
        
        # Check nucleotide composition
        invalid_chars = set(sequence) - self.valid_nucleotides
        if invalid_chars:
            return False, f"Invalid characters: {invalid_chars}"
        
        # Check for excessive homopolymers (>20 consecutive same nucleotide)
        for nucleotide in self.valid_nucleotides:
            if nucleotide * 20 in sequence:
                return False, f"Excessive homopolymer: {nucleotide}"
        
        # Check GC content (should be 30-70%)
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        if gc_content < 0.3 or gc_content > 0.7:
            return False, f"Abnormal GC content: {gc_content:.2f}"
        
        return True, "Valid"
    
    def process_fasta_file(
        self, 
        input_file: Path, 
        output_file: Path,
        label: str
    ) -> pd.DataFrame:
        """
        Process a FASTA file and save cleaned sequences
        
        Args:
            input_file: Input FASTA file path
            output_file: Output FASTA file path
            label: Class label for sequences
            
        Returns:
            DataFrame with processed sequence metadata
        """
        metadata = []
        valid_records = []
        
        logger.info(f"Processing {input_file}")
        
        for record in SeqIO.parse(input_file, "fasta"):
            # Clean sequence
            cleaned_seq = self.clean_sequence(str(record.seq))
            
            # Validate
            is_valid, reason = self.validate_sequence(cleaned_seq)
            
            if is_valid:
                # Update record with cleaned sequence
                record.seq = Seq(cleaned_seq)
                valid_records.append(record)
                
                # Calculate sequence statistics
                gc_content = (cleaned_seq.count('G') + cleaned_seq.count('C')) / len(cleaned_seq)
                
                metadata.append({
                    'id': record.id,
                    'label': label,
                    'length': len(cleaned_seq),
                    'gc_content': gc_content,
                    'description': record.description
                })
            else:
                logger.debug(f"Skipped {record.id}: {reason}")
        
        # Save valid sequences
        output_file.parent.mkdir(parents=True, exist_ok=True)
        SeqIO.write(valid_records, output_file, "fasta")
        
        logger.info(f"Saved {len(valid_records)} valid sequences to {output_file}")
        
        return pd.DataFrame(metadata)
    
    def create_train_val_test_splits(
        self,
        metadata_df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        balance_classes: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Create stratified train/val/test splits
        
        Args:
            metadata_df: DataFrame with sequence metadata
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            balance_classes: Whether to balance class distribution
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Balance classes if requested
        if balance_classes:
            min_samples = metadata_df['label'].value_counts().min()
            balanced_dfs = []
            
            for label in metadata_df['label'].unique():
                label_df = metadata_df[metadata_df['label'] == label]
                balanced_dfs.append(
                    label_df.sample(n=min_samples, random_state=random_seed)
                )
            
            metadata_df = pd.concat(balanced_dfs, ignore_index=True)
            logger.info(f"Balanced dataset: {min_samples} samples per class")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            metadata_df,
            test_size=(val_ratio + test_ratio),
            stratify=metadata_df['label'],
            random_state=random_seed
        )
        
        # Second split: val vs test
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size_adjusted),
            stratify=temp_df['label'],
            random_state=random_seed
        )
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        # Log split statistics
        for split_name, split_df in splits.items():
            logger.info(f"\n{split_name.upper()} Split:")
            logger.info(f"  Total samples: {len(split_df)}")
            logger.info(f"  Class distribution:\n{split_df['label'].value_counts()}")
        
        return splits