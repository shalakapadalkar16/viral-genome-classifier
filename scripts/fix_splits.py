import pandas as pd
from pathlib import Path

# Load existing splits
splits_dir = Path('data/splits')
processed_dir = Path('data/processed')

for split_name in ['train', 'val', 'test']:
    split_df = pd.read_csv(splits_dir / f'{split_name}.csv')
    
    # Add file path for each sequence based on label
    split_df['file'] = split_df['label'].apply(
        lambda x: str(processed_dir / f"{x.lower()}_clean.fasta")
    )
    
    # Save updated split
    split_df.to_csv(splits_dir / f'{split_name}.csv', index=False)
    print(f"Updated {split_name}.csv with file paths")

print("\n✅ All splits updated!")