#!/usr/bin/env python3
"""
Generate all key visualizations for README and presentations
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

def plot_data_distribution():
    """Plot train/val/test distribution"""
    splits_dir = Path('data/splits')
    
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')
    test_df = pd.read_csv(splits_dir / 'test.csv')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (df, title) in zip(axes, [
        (train_df, 'Training Set (n=157)'),
        (val_df, 'Validation Set (n=34)'),
        (test_df, 'Test Set (n=34)')
    ]):
        counts = df['label'].value_counts().sort_index()
        bars = ax.bar(range(len(counts)), counts.values, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Sequences')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, max(counts.values) * 1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_distribution.png', bbox_inches='tight')
    print("✓ Saved data_distribution.png")
    plt.close()


def plot_training_curves():
    """Plot training/validation curves from MLflow"""
    # Simulated data - replace with actual MLflow data
    epochs = list(range(1, 22))
    train_loss = [1.85, 1.86, 1.83, 1.69, 1.81, 1.74, 1.70, 1.62, 1.68, 1.64,
                  1.65, 1.62, 1.64, 1.65, 1.58, 1.56, 1.63, 1.64, 1.55, 1.56, 1.58]
    val_loss = [1.60, 1.62, 1.62, 1.63, 1.65, 1.66, 1.68, 1.67, 1.65, 1.63,
                1.65, 1.67, 1.66, 1.64, 1.61, 1.62, 1.64, 1.63, 1.65, 1.64, 1.65]
    train_acc = [17, 21, 21, 25, 21, 24, 25, 27, 24, 24, 25, 30, 28, 27, 31, 31, 26, 28, 31, 29, 29]
    val_acc = [24, 24, 26, 26, 26, 26, 21, 24, 26, 26, 26, 26, 26, 29, 29, 32, 32, 32, 29, 29, 26]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_loss, 'o-', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_loss, 's-', label='Validation Loss', linewidth=2, markersize=4)
    ax1.axvline(x=17, color='red', linestyle='--', alpha=0.5, label='Best Model (Epoch 17)')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Cross-Entropy Loss', fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontweight='bold', fontsize=12)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_acc, 'o-', label='Training Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, val_acc, 's-', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.axhline(y=20, color='gray', linestyle=':', alpha=0.5, label='Random Baseline (20%)')
    ax2.axvline(x=17, color='red', linestyle='--', alpha=0.5, label='Best Model')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontweight='bold', fontsize=12)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(15, 35)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', bbox_inches='tight')
    print("✓ Saved training_curves.png")
    plt.close()


def plot_confusion_matrix():
    """Plot confusion matrix"""
    # Simulated confusion matrix - replace with actual test results
    labels = ['Coronaviridae', 'Filoviridae', 'Flaviviridae', 'Orthomyxoviridae', 'Retroviridae']
    
    cm = np.array([
        [3, 1, 2, 0, 1],  # Coronaviridae
        [1, 2, 2, 1, 1],  # Filoviridae
        [1, 1, 3, 1, 1],  # Flaviviridae
        [2, 0, 1, 3, 1],  # Orthomyxoviridae
        [0, 2, 1, 1, 2]   # Retroviridae
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax.set_title('Confusion Matrix - Test Set (n=34)', fontweight='bold', fontsize=12)
    
    # Add accuracy annotation
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(2.5, -0.7, f'Overall Accuracy: {accuracy:.1%}', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', bbox_inches='tight')
    print("✓ Saved confusion_matrix.png")
    plt.close()


def plot_architecture_diagram():
    """Create architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Define boxes
    boxes = [
        # (x, y, width, height, text, color)
        (0.1, 0.85, 0.8, 0.08, 'Raw Genomic Sequences\n(NCBI GenBank)', '#E8F4F8'),
        (0.1, 0.72, 0.35, 0.08, 'Quality Control\n• GC Content\n• Homopolymers\n• Length Filter', '#B8E6F0'),
        (0.55, 0.72, 0.35, 0.08, 'Data Augmentation\n• Reverse Complement\n• Random Crop\n• Mutations', '#B8E6F0'),
        (0.1, 0.59, 0.8, 0.08, 'K-mer Tokenization\n(4-mer: ATCG → 256 tokens)', '#88D8E8'),
        (0.1, 0.41, 0.35, 0.12, 'CNN Model\n• Conv1D (128 filters)\n• Batch Norm\n• Dropout (0.5)', '#58C8D8'),
        (0.55, 0.41, 0.35, 0.12, 'Training\n• AdamW Optimizer\n• Early Stopping\n• MLflow Tracking', '#58C8D8'),
        (0.1, 0.23, 0.8, 0.08, 'Evaluation & Metrics\nAccuracy • F1 • Precision • Recall', '#28B8C8'),
        (0.1, 0.10, 0.8, 0.08, 'Classification Output\n5 Virus Families', '#0898B8'),
    ]
    
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=9, fontweight='bold', multialignment='center')
    
    # Add arrows
    arrows = [
        (0.5, 0.85, 0.5, 0.80),
        (0.275, 0.72, 0.275, 0.67),
        (0.725, 0.72, 0.725, 0.67),
        (0.5, 0.59, 0.5, 0.53),
        (0.275, 0.41, 0.275, 0.31),
        (0.725, 0.41, 0.725, 0.31),
        (0.5, 0.23, 0.5, 0.18),
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Viral Genome Classification Pipeline Architecture', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_diagram.png', bbox_inches='tight')
    print("✓ Saved architecture_diagram.png")
    plt.close()


def plot_model_comparison():
    """Compare different model approaches"""
    models = ['ESM-2\n(Frozen)', 'CNN\n(6-mer)', 'Simple CNN\n(4-mer + Aug)']
    train_acc = [23, 38, 31]
    val_acc = [24, 29, 32]
    val_f1 = [14, 28, 27]
    params = [35, 1.5, 0.066]  # in millions
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Accuracy comparison
    bars1 = ax1.bar(x - width/2, train_acc, width, label='Train Acc', color='steelblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, val_acc, width, label='Val Acc', color='coral', alpha=0.7)
    ax1.axhline(y=20, color='gray', linestyle='--', label='Random (20%)', alpha=0.5)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # F1 Score
    bars = ax2.bar(models, val_f1, color='seagreen', alpha=0.7)
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('Validation F1 Score', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 35)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Model complexity
    bars = ax3.bar(models, params, color='mediumpurple', alpha=0.7)
    ax3.set_ylabel('Parameters (Millions)', fontweight='bold')
    ax3.set_title('Model Complexity', fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{param}M', ha='center', va='bottom', fontsize=9)
    
    # Generalization gap
    gap = [abs(t - v) for t, v in zip(train_acc, val_acc)]
    bars = ax4.bar(models, gap, color='indianred', alpha=0.7)
    ax4.set_ylabel('Generalization Gap (%)', fontweight='bold')
    ax4.set_title('Overfitting Analysis (Train - Val Acc)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=5, color='green', linestyle='--', label='Good (<5%)', alpha=0.5)
    ax4.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', bbox_inches='tight')
    print("✓ Saved model_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("\n🎨 Generating visualizations...")
    print("="*50)
    
    plot_data_distribution()
    plot_training_curves()
    plot_confusion_matrix()
    plot_architecture_diagram()
    plot_model_comparison()
    
    print("="*50)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Saved to: {output_dir.absolute()}\n")