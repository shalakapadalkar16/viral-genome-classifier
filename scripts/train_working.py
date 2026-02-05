#!/usr/bin/env python3
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

from src.data.augmented_dataset import AugmentedViralGenomeDataset
from src.data.simple_tokenization import Simple4merTokenizer
from src.models.simple_cnn import SimpleCNNClassifier
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/working_config.yaml")
    parser.add_argument("--experiment-name", type=str, default="viral-genome-working")
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🧬 Training with Data Augmentation")
    logger.info("="*60)
    
    config = load_config(args.config)
    device = config['training']['device']
    
    torch.manual_seed(config['project']['random_seed'])
    
    mlflow.set_tracking_uri(config['logging']['mlflow_tracking_uri'])
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        # Initialize tokenizer
        logger.info("Initializing Simple 4-mer tokenizer...")
        tokenizer = Simple4merTokenizer()
        logger.info(f"✅ Vocabulary size: {tokenizer.vocab_size}")
        
        # Load data
        splits_dir = Path(config['data']['splits_dir'])
        train_df = pd.read_csv(splits_dir / 'train.csv')
        val_df = pd.read_csv(splits_dir / 'val.csv')
        
        unique_labels = sorted(train_df['label'].unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Create datasets WITH AUGMENTATION
        logger.info("Creating training dataset with augmentation...")
        train_dataset = AugmentedViralGenomeDataset(
            train_df,
            tokenizer,
            max_length=config['model']['max_tokens'],
            label_to_id=label_to_id,
            augment=config['data']['augment_training'],
            augmentation_factor=config['data']['augmentation_factor']
        )
        
        logger.info("Creating validation dataset...")
        val_dataset = AugmentedViralGenomeDataset(
            val_df,
            tokenizer,
            max_length=config['model']['max_tokens'],
            label_to_id=label_to_id,
            augment=False  # No augmentation for validation
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        # Initialize model
        logger.info("Initializing Simple CNN model...")
        model = SimpleCNNClassifier(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=config['model']['embedding_dim'],
            num_labels=len(label_to_id),
            dropout=config['model']['dropout']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Train
        trainer = GenomeClassifierTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=config,
            device=device
        )
        
        logger.info("\n🚀 Starting Training with Augmented Data...\n")
        
        try:
            final_metrics = trainer.train()
            
            logger.info("\n" + "="*60)
            logger.info("✅ Training Complete!")
            logger.info("="*60)
            logger.info(f"Best val F1: {trainer.best_val_metric:.4f}")
            logger.info(f"Best val accuracy: {final_metrics['accuracy']:.4f}")
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted")
        except Exception as e:
            logger.error(f"\n❌ Failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()