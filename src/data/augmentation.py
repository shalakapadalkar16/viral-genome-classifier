"""
Data augmentation for DNA sequences
"""
import random
from typing import List
from Bio.Seq import Seq


class DNASequenceAugmenter:
    """Augment DNA sequences for training"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def reverse_complement(self, sequence: str) -> str:
        """Get reverse complement of sequence"""
        seq_obj = Seq(sequence)
        return str(seq_obj.reverse_complement())
    
    def random_crop(self, sequence: str, crop_ratio: float = 0.9) -> str:
        """Randomly crop sequence"""
        if len(sequence) < 100:
            return sequence
        
        crop_len = int(len(sequence) * crop_ratio)
        max_start = len(sequence) - crop_len
        start = random.randint(0, max_start)
        return sequence[start:start + crop_len]
    
    def add_noise(self, sequence: str, noise_rate: float = 0.01) -> str:
        """Add random mutations to sequence"""
        seq_list = list(sequence)
        bases = ['A', 'T', 'C', 'G']
        
        for i in range(len(seq_list)):
            if random.random() < noise_rate:
                seq_list[i] = random.choice(bases)
        
        return ''.join(seq_list)
    
    def augment(self, sequence: str, num_augments: int = 2) -> List[str]:
        """
        Generate augmented versions of sequence
        
        Args:
            sequence: Original DNA sequence
            num_augments: Number of augmented versions to create
            
        Returns:
            List of augmented sequences (includes original)
        """
        augmented = [sequence]  # Always include original
        
        methods = [
            lambda s: self.reverse_complement(s),
            lambda s: self.random_crop(s, 0.95),
            lambda s: self.random_crop(s, 0.90),
            lambda s: self.add_noise(s, 0.005),
            lambda s: self.add_noise(s, 0.01),
        ]
        
        # Randomly select augmentation methods
        for _ in range(min(num_augments, len(methods))):
            method = random.choice(methods)
            aug_seq = method(sequence)
            augmented.append(aug_seq)
        
        return augmented[:num_augments + 1]