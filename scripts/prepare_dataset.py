"""
Main script to download and prepare dataset
"""
import os
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.download import NCBIGenomeDownloader
from data.preprocessing import GenomePreprocessor


def main():
    parser = argparse.ArgumentParser(description="Download and prepare viral genome dataset")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--download", action="store_true",
                       help="Download sequences from NCBI")
    parser.add_argument("--process", action="store_true",
                       help="Process and clean sequences")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    load_dotenv()
    
    if args.download:
        # Initialize downloader
        downloader = NCBIGenomeDownloader(
            email=os.getenv("NCBI_EMAIL"),
            api_key=os.getenv("NCBI_API_KEY"),
            output_dir=config['data']['raw_dir']
        )
        
        # Download all families
        metadata = downloader.download_all_families(
            families=config['data']['target_families'],
            samples_per_family=config['data']['samples_per_family']
        )
        
        print(f"\nDownload complete! Total sequences: {len(metadata)}")
    
    if args.process:
        # Initialize preprocessor
        preprocessor = GenomePreprocessor(
            min_length=config['data']['min_sequence_length'],
            max_length=config['data']['max_sequence_length']
        )
        
        # Process each family
        raw_dir = Path(config['data']['raw_dir'])
        processed_dir = Path(config['data']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        all_metadata = []
        
        for family in config['data']['target_families']:
            input_file = raw_dir / f"{family.lower()}.fasta"
            output_file = processed_dir / f"{family.lower()}_clean.fasta"
            
            if input_file.exists():
                metadata = preprocessor.process_fasta_file(
                    input_file=input_file,
                    output_file=output_file,
                    label=family
                )
                all_metadata.append(metadata)
        
        # Combine and create splits
        if all_metadata:
            import pandas as pd
            combined_metadata = pd.concat(all_metadata, ignore_index=True)
            
            splits = preprocessor.create_train_val_test_splits(
                combined_metadata,
                train_ratio=config['data']['train_ratio'],
                val_ratio=config['data']['val_ratio'],
                test_ratio=config['data']['test_ratio'],
                balance_classes=config['data']['balance_classes']
            )
            
            # Save splits
            splits_dir = Path(config['data']['splits_dir'])
            splits_dir.mkdir(parents=True, exist_ok=True)
            
            for split_name, split_df in splits.items():
                split_df.to_csv(splits_dir / f"{split_name}.csv", index=False)
            
            print("\nDataset preparation complete!")
            print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    main()