"""
Training loop for genome classification
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import mlflow
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class GenomeClassifierTrainer:
    """Trainer for viral genome classification"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = self._setup_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        
        # Tracking
        self.best_val_metric = 0.0
        self.global_step = 0
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with weight decay"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['training']['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate']
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        num_training_steps = len(self.train_dataloader) * self.config['training']['num_epochs']
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs['logits'], dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
            
            # Log to MLflow
            if self.global_step % 10 == 0:
                mlflow.log_metric("train_loss", loss.item(), step=self.global_step)
                mlflow.log_metric("learning_rate", self.scheduler.get_last_lr()[0], step=self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask, labels)
            
            total_loss += outputs['loss'].item()
            predictions = torch.argmax(outputs['logits'], dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_dataloader)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self) -> Dict[str, float]:
        """Full training loop with early stopping"""
        logger.info("Starting training...")
        
        # Early stopping setup
        patience = self.config['training'].get('patience', 5)
        min_delta = self.config['training'].get('min_delta', 0.001)
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            logger.info(f"\nEpoch {epoch}/{self.config['training']['num_epochs']}")
            
            # Unfreeze encoder if specified
            if (hasattr(self.model, 'unfreeze_encoder') and 
                epoch == self.config['model'].get('unfreeze_after_epoch', -1)):
                self.model.unfreeze_encoder()
                logger.info("🔓 Unfreezing encoder for fine-tuning!")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate()
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}, "
                       f"Val F1: {val_metrics['f1']:.4f}")
            
            # Log to MLflow
            mlflow.log_metrics({
                f"train_{k}": v for k, v in train_metrics.items()
            }, step=epoch)
            mlflow.log_metrics({
                f"val_{k}": v for k, v in val_metrics.items()
            }, step=epoch)
            
            # Save best model
            if val_metrics['f1'] > self.best_val_metric:
                self.best_val_metric = val_metrics['f1']
                self.save_checkpoint(epoch, val_metrics['f1'])
                logger.info(f"✨ New best model! F1: {val_metrics['f1']:.4f}")
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss - min_delta:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"⏸️  Early stopping patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    logger.info(f"⛔ Early stopping triggered at epoch {epoch}")
                    break
        
        logger.info("\n✅ Training complete!")
        return val_metrics
    
    def save_checkpoint(self, epoch: int, metric: float):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_f1{metric:.4f}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric': metric,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")