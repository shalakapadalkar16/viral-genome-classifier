#!/usr/bin/env python3
"""
Training script for CNN model with k-mer tokenization
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ViralGenomeDataset
from src.data.tokenization import DNAKmerTokenizer
from src.models.cnn_classifier import DNACNNClassifier
from src.training.trainer import GenomeClassifierTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, tokenizer):
    splits_dir = Path(config['data']['splits_dir'])
    
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')
    
    logger.info(f"Loaded train split: {len(train_df)} samples")
    logger.info(f"Loaded val split: {len(val_df)} samples")
    
    # Create label mapping
    unique_labels = sorted(train_df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    
    logger.info(f"Label mapping: {label_to_id}")
    
    # Create datasets
    train_dataset = ViralGenomeDataset(
        train_df,
        tokenizer,
        max_length=config['model']['max_tokens'],
        label_to_id=label_to_id
    )
    
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
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    return train_loader, val_loader, label_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cnn_config.yaml")
    parser.add_argument("--experiment-name", type=str, default="viral-genome-cnn")
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🧬 Training DNA CNN Classifier")
    logger.info("="*60)
    
    config = load_config(args.config)
    
    device = config['training'].get('device', 'cpu')
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(config['project']['random_seed'])
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['logging']['mlflow_tracking_uri'])
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        # Initialize k-mer tokenizer
        logger.info("Initializing DNA K-mer tokenizer (k=6)...")
        tokenizer = DNAKmerTokenizer(k=6)
        logger.info(f"✅ Vocabulary size: {tokenizer.vocab_size}")
        
        # Create dataloaders
        train_loader, val_loader, label_to_id = create_dataloaders(config, tokenizer)
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        # Initialize model
        logger.info("Initializing CNN model...")
        model = DNACNNClassifier(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=config['model']['embedding_dim'],
            num_labels=len(label_to_id),
            dropout=config['model']['dropout']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Initialize trainer
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
        
        try:
            final_metrics = trainer.train()
            
            logger.info("\n" + "="*60)
            logger.info("✅ Training Complete!")
            logger.info("="*60)
            logger.info(f"Final val F1: {final_metrics['f1']:.4f}")
            logger.info(f"Final val accuracy: {final_metrics['accuracy']:.4f}")
            logger.info(f"Best val F1: {trainer.best_val_metric:.4f}")
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  Training interrupted")
        except Exception as e:
            logger.error(f"\n❌ Training failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()