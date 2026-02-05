"""
Evaluation script for trained model
"""
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path
import sys
from torch.utils.data import DataLoader
from transformers import EsmTokenizer

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import ViralGenomeDataset
from models.esm_classifier import ESM2Classifier
from evaluation.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    test_df = pd.read_csv(Path(config['data']['splits_dir']) / 'test.csv')
    
    # Initialize tokenizer
    tokenizer = EsmTokenizer.from_pretrained(config['model']['pretrained_model'])
    
    # Create label mapping
    unique_labels = sorted(test_df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    # Create dataset
    test_dataset = ViralGenomeDataset(
        test_df,
        tokenizer,
        max_length=config['model']['max_tokens'],
        label_to_id=label_to_id
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = ESM2Classifier(
        model_name=config['model']['pretrained_model'],
        num_labels=len(label_to_id),
        hidden_dropout_prob=config['model']['classifier_dropout'],
        hidden_dims=config['model']['hidden_dims']
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config['training']['device'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(id_to_label=id_to_label)
    
    # Evaluate
    results = evaluator.evaluate_model(
        model,
        test_loader,
        device=config['training']['device']
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nAccuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Weighted F1: {results['metrics']['f1_weighted']:.4f}")
    print(f"Macro F1: {results['metrics']['f1_macro']:.4f}")
    
    if results['metrics']['roc_auc']:
        print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
    
    print("\nPer-class metrics:")
    for label, metrics in results['metrics']['per_class'].items():
        print(f"\n{label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Support: {metrics['support']}")


if __name__ == "__main__":
    main()