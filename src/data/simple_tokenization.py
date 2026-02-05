"""
Simpler 4-mer tokenization (smaller vocabulary)
"""
from typing import List
import torch


class Simple4merTokenizer:
    """Simple 4-mer tokenizer with small vocabulary"""
    
    def __init__(self):
        self.k = 4
        
        # Generate all 4-mers (256 total)
        bases = ['A', 'T', 'C', 'G']
        self.vocab = []
        for b1 in bases:
            for b2 in bases:
                for b3 in bases:
                    for b4 in bases:
                        self.vocab.append(b1 + b2 + b3 + b4)
        
        self.token_to_id = {token: idx + 2 for idx, token in enumerate(self.vocab)}
        self.token_to_id['<PAD>'] = 0
        self.token_to_id['<CLS>'] = 1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def tokenize(self, sequence: str) -> List[str]:
        """Convert sequence to 4-mers"""
        sequence = sequence.upper()
        kmers = ['<CLS>']
        
        for i in range(0, len(sequence) - self.k + 1, 2):  # Step by 2 for efficiency
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
        return_tensors: str = None
    ) -> dict:
        """Encode sequence to token IDs"""
        kmers = self.tokenize(sequence)
        input_ids = [self.token_to_id.get(kmer, 0) for kmer in kmers]
        
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        attention_mask = [1] * len(input_ids)
        
        if padding == 'max_length':
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor([result['input_ids']])
            result['attention_mask'] = torch.tensor([result['attention_mask']])
        
        return result
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
    
    @property
    def vocab_size(self):
        return len(self.token_to_id)