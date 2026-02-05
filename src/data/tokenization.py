"""
K-mer tokenization for DNA sequences
"""
from typing import List
import torch


class DNAKmerTokenizer:
    """Convert DNA sequences to k-mer tokens"""
    
    def __init__(self, k: int = 6):
        """
        Initialize k-mer tokenizer
        
        Args:
            k: K-mer size (default 6)
        """
        self.k = k
        
        # Create vocabulary of all possible k-mers
        bases = ['A', 'T', 'C', 'G']
        self.vocab = self._generate_kmers(bases, k)
        self.token_to_id = {token: idx + 2 for idx, token in enumerate(self.vocab)}
        self.token_to_id['<PAD>'] = 0
        self.token_to_id['<CLS>'] = 1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
    def _generate_kmers(self, bases: List[str], k: int) -> List[str]:
        """Generate all possible k-mers"""
        if k == 1:
            return bases
        
        smaller_kmers = self._generate_kmers(bases, k - 1)
        kmers = []
        for base in bases:
            for kmer in smaller_kmers:
                kmers.append(base + kmer)
        return kmers
    
    def tokenize(self, sequence: str) -> List[str]:
        """
        Convert sequence to k-mers
        
        Args:
            sequence: DNA sequence
            
        Returns:
            List of k-mer tokens
        """
        sequence = sequence.upper()
        kmers = ['<CLS>']
        
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            if all(base in 'ATCG' for base in kmer):
                kmers.append(kmer)
        
        return kmers
    
    def encode(
        self,
        sequence: str,
        max_length: int = 512,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = None  # ADD THIS PARAMETER
    ) -> dict:
        """
        Encode sequence to token IDs
        
        Args:
            sequence: DNA sequence
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: 'pt' for PyTorch tensors, None for lists
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Get k-mers
        kmers = self.tokenize(sequence)
        
        # Convert to IDs
        input_ids = [self.token_to_id.get(kmer, 0) for kmer in kmers]
        
        # Truncate if needed
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if padding == 'max_length':
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor([result['input_ids']])
            result['attention_mask'] = torch.tensor([result['attention_mask']])
        
        return result
    
    def __call__(self, *args, **kwargs):
        """Make tokenizer callable"""
        return self.encode(*args, **kwargs)
    
    @property
    def vocab_size(self):
        """Return vocabulary size"""
        return len(self.token_to_id)