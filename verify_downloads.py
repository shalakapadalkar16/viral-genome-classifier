# Quick verification script
import pandas as pd
from Bio import SeqIO

metadata = pd.read_csv('data/raw/all_sequences_metadata.csv')
print(f"Total sequences: {len(metadata)}")
print(f"\nPer family:")
print(metadata['family'].value_counts())

# Check file sizes
for family in metadata['family'].unique():
    file_path = f"data/raw/{family.lower()}.fasta"
    seqs = list(SeqIO.parse(file_path, "fasta"))
    print(f"{family}: {len(seqs)} sequences")