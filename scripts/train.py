#!/usr/bin/env python3
"""
Main training script for viral genome classification
"""
import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import mlflow
from transformers import EsmTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ViralGenomeDataset
from src.models.esm_classifier import ESM2Classifier
from src.training.trainer import GenomeClassifierTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, tokenizer):
    """Create train/val/test dataloaders"""
    splits_dir = Path(config['data']['splits_dir'])
    
    # Load split metadata
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')
    
    logger.info(f"Loaded train split: {len(train_df)} samples")
    logger.info(f"Loaded val split: {len(val_df)} samples")
    
    # Verify 'file' column exists
    if 'file' not in train_df.columns:
        logger.error("'file' column missing from splits!")
        logger.info("Running fix_splits.py to add file paths...")
        
        # Add file paths
        processed_dir = Path(config['data'].get('processed_dir', 'data/processed'))
        train_df['file'] = train_df['label'].apply(
            lambda x: str(processed_dir / f"{x.lower()}_clean.fasta")
        )
        val_df['file'] = val_df['label'].apply(
            lambda x: str(processed_dir / f"{x.lower()}_clean.fasta")
        )
        
        # Save updated splits
        train_df.to_csv(splits_dir / 'train.csv', index=False)
        val_df.to_csv(splits_dir / 'val.csv', index=False)
        logger.info("✅ Fixed splits with file paths")
    
    # Create label mapping
    unique_labels = sorted(train_df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    
    logger.info(f"Label mapping: {label_to_id}")
    
    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = ViralGenomeDataset(
        train_df,
        tokenizer,
        max_length=config['model']['max_tokens'],
        label_to_id=label_to_id
    )
    
    logger.info("Creating validation dataset...")
    val_dataset = ViralGenomeDataset(
        val_df,
        tokenizer,
        max_length=config['model']['max_tokens'],
        label_to_id=label_to_id
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=False  # Set to False for Mac
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=False
    )
    
    return train_loader, val_loader, label_to_id


def main():
    parser = argparse.ArgumentParser(description="Train viral genome classifier")
    parser.add_argument("--config", type=str, default="configs/small_dataset_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, default="viral-genome-classification",
                       help="MLflow experiment name")
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🧬 Viral Genome Classification Training")
    logger.info("="*60)
    
    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Check device
    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
        config['training']['device'] = 'cpu'
        config['training']['mixed_precision'] = False  # Disable for CPU
    
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['project']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['project']['random_seed'])
    
    # Setup MLflow
    mlflow_uri = config['logging']['mlflow_tracking_uri']
    logger.info(f"Setting up MLflow at {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log config
        mlflow.log_params(flatten_dict(config))
        
        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = EsmTokenizer.from_pretrained(
            config['model']['pretrained_model']
        )
        logger.info(f"✅ Loaded tokenizer: {config['model']['pretrained_model']}")
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, label_to_id = create_dataloaders(config, tokenizer)
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Number of classes: {len(label_to_id)}")
        
        # Initialize model
        logger.info("Initializing model...")
        model = ESM2Classifier(
            model_name=config['model']['pretrained_model'],
            num_labels=len(label_to_id),
            hidden_dropout_prob=config['model']['classifier_dropout'],
            freeze_encoder=config['model']['freeze_encoder'],
            hidden_dims=config['model']['hidden_dims']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        
        # Log model info
        mlflow.log_param("total_params", total_params)
        mlflow.log_param("trainable_params", trainable_params)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = GenomeClassifierTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=config,
            device=device
        )
        
        logger.info("\n" + "="*60)
        logger.info("🚀 Starting Training...")
        logger.info("="*60 + "\n")
        
        # Train
        try:
            final_metrics = trainer.train()
            
            # Log final metrics
            mlflow.log_metrics({
                f"final_{k}": v for k, v in final_metrics.items()
            })
            
            logger.info("\n" + "="*60)
            logger.info("✅ Training Complete!")
            logger.info("="*60)
            logger.info(f"Final validation F1: {final_metrics['f1']:.4f}")
            logger.info(f"Final validation accuracy: {final_metrics['accuracy']:.4f}")
            logger.info(f"Best validation F1: {trainer.best_val_metric:.4f}")
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  Training interrupted by user")
            mlflow.set_tag("status", "interrupted")
        except Exception as e:
            logger.error(f"\n❌ Training failed with error: {str(e)}")
            logger.exception("Full traceback:")
            mlflow.set_tag("status", "failed")
            raise


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary for MLflow logging"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    main()