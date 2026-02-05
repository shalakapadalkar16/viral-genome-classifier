"""
Comprehensive model evaluation framework
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluation for genome classification"""
    
    def __init__(self, id_to_label: Dict[int, str], output_dir: str = "results"):
        """
        Initialize evaluator
        
        Args:
            id_to_label: Mapping from label IDs to names
            output_dir: Directory to save results
        """
        self.id_to_label = id_to_label
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            dataloader: Test data loader
            device: Device to run on
            
        Returns:
            Dictionary with all evaluation metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_sequence_ids = []
        
        logger.info("Running inference on test set...")
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            
            # Get predictions and probabilities
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_sequence_ids.extend(batch['sequence_id'])
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Generate visualizations
        self._plot_confusion_matrix(all_labels, all_predictions)
        self._plot_roc_curves(all_labels, all_probabilities)
        self._plot_class_distribution(all_labels, all_predictions)
        
        # Generate detailed report
        report = self._generate_report(all_labels, all_predictions, all_probabilities)
        
        # Save predictions
        self._save_predictions(all_sequence_ids, all_labels, all_predictions, all_probabilities)
        
        logger.info("Evaluation complete!")
        return {
            'metrics': metrics,
            'report': report
        }
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """Calculate all evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(
                labels,
                probabilities,
                multi_class='ovr',
                average='weighted'
            )
        except:
            roc_auc = None
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'per_class': {
                self.id_to_label[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(precision))
            }
        }
        
        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _plot_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.id_to_label[i] for i in range(len(cm))],
            yticklabels=[self.id_to_label[i] for i in range(len(cm))]
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def _plot_roc_curves(self, labels: np.ndarray, probabilities: np.ndarray):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        n_classes = probabilities.shape[1]
        labels_bin = label_binarize(labels, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
            auc = roc_auc_score(labels_bin[:, i], probabilities[:, i])
            
            plt.plot(fpr, tpr, label=f'{self.id_to_label[i]} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'roc_curves.png', dpi=300)
        plt.close()
    
    def _plot_class_distribution(self, labels: np.ndarray, predictions: np.ndarray):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # True distribution
        true_counts = pd.Series(labels).value_counts().sort_index()
        true_labels = [self.id_to_label[i] for i in true_counts.index]
        ax1.bar(true_labels, true_counts.values)
        ax1.set_title('True Label Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted distribution
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        pred_labels = [self.id_to_label[i] for i in pred_counts.index]
        ax2.bar(pred_labels, pred_counts.values, color='orange')
        ax2.set_title('Predicted Label Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'class_distribution.png', dpi=300)
        plt.close()
    
    def _generate_report(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> str:
        """Generate detailed classification report"""
        target_names = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        
        report = classification_report(
            labels,
            predictions,
            target_names=target_names,
            digits=4
        )
        
        # Save report
        with open(self.output_dir / 'reports' / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        return report
    
    def _save_predictions(
        self,
        sequence_ids: List[str],
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ):
        """Save predictions to CSV"""
        results_df = pd.DataFrame({
            'sequence_id': sequence_ids,
            'true_label': [self.id_to_label[l] for l in labels],
            'predicted_label': [self.id_to_label[p] for p in predictions],
            'correct': labels == predictions
        })
        
        # Add probability columns
        for i in range(probabilities.shape[1]):
            results_df[f'prob_{self.id_to_label[i]}'] = probabilities[:, i]
        
        results_df.to_csv(
            self.output_dir / 'predictions' / 'test_predictions.csv',
            index=False
        )