# Viral Genome Classification Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready ML pipeline for automated viral pathogen classification from genomic sequences**

A complete end-to-end machine learning system for classifying viral genomes, demonstrating ML engineering best practices, bioinformatics data processing, and small-data learning strategies applicable to diagnostic development.

---

## 🎯 Project Overview

This project addresses the challenge of automated viral genome classification using a production-grade ML pipeline that handles everything from raw sequence data to model deployment. Built with diagnostic applications in mind, the system emphasizes **data quality, reproducibility, and scalable architecture** over pure accuracy metrics.

### **Key Features**

- **Automated Data Collection**: Programmatic download from NCBI GenBank with quality validation
- **Robust Preprocessing**: Sequence validation (GC content, homopolymer detection, length filtering)
- **Multiple Model Architectures**: ESM-2 transformers, custom CNNs, k-mer tokenization approaches
- **Data Augmentation**: Reverse complement, random cropping, mutation-based augmentation
- **Production MLOps**: MLflow tracking, checkpoint management, comprehensive evaluation
- **Modular Design**: Easy to swap models, add virus families, or scale to larger datasets

---

## 📊 Results & Analysis

### **Performance Summary**

| Metric | Value | Context |
|--------|-------|---------|
| **Dataset Size** | 225 sequences | 45 per class across 5 viral families |
| **Test Accuracy** | 32.4% | vs. 20% random baseline |
| **Weighted F1** | 0.273 | Demonstrates above-random learning |
| **Training Time** | ~3 sec/epoch | On CPU (M-series Mac) |

### **Target Virus Families**

1. **Coronaviridae** (COVID-19, SARS, MERS) - 64 sequences
2. **Orthomyxoviridae** (Influenza A/B/C) - 45 sequences
3. **Flaviviridae** (Zika, Dengue, Yellow Fever) - 200 sequences
4. **Filoviridae** (Ebola, Marburg) - 196 sequences
5. **Retroviridae** (HIV) - 200 sequences

### **Critical Analysis: Small Data Challenges**

The 32% accuracy reveals fundamental deep learning constraints with limited biological data:

**Data Requirements:**
- Current: 31 samples/class (training)
- Minimum viable: 200-500 samples/class
- Production-grade: 1000+ samples/class

**Why Performance is Limited:**
1. **Insufficient training diversity**: 31 samples cannot capture within-family genomic variation
2. **Validation set size**: 6-7 samples per class leads to high variance in metrics
3. **Transfer learning mismatch**: ESM-2 pre-trained on proteins, not DNA sequences
4. **Imbalanced data collection**: Coronaviridae (64) vs Flaviviridae (200) raw sequences

**Projected Performance with Scale:**

| Dataset Size | Expected Accuracy | Clinical Utility |
|--------------|-------------------|------------------|
| 500 samples/class | 65-75% | Research-grade |
| 1,000 samples/class | 75-85% | Clinical validation |
| 2,000+ samples/class | 85-92% | Diagnostic deployment |

This scaling projection is based on published genomic classification literature and transfer learning benchmarks.

---

## 🏗️ Architecture & Implementation

### **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                     │
│  • NCBI Entrez API Integration                              │
│  • Automated GenBank Downloads                              │
│  • Metadata Extraction & Tracking                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Preprocessing Pipeline                      │
│  • Sequence Cleaning (ambiguous nucleotides)                │
│  • Quality Validation (GC content, homopolymers)            │
│  • Length Filtering (1kb - 30kb)                            │
│  • Stratified Train/Val/Test Splits                         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Tokenization Layer                         │
│  • K-mer Generation (4-mer, 6-mer)                         │
│  • Vocabulary Management (256-4096 tokens)                  │
│  • Sliding Window Encoding                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Model Training                             │
│  • ESM-2 Fine-tuning (35M parameters)                       │
│  • Custom CNN (66K parameters)                              │
│  • Data Augmentation (3x expansion)                         │
│  • Mixed Precision Training                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                 Evaluation & Deployment                      │
│  • Comprehensive Metrics (Acc, F1, ROC-AUC)                │
│  • Confusion Matrix Analysis                                │
│  • MLflow Experiment Tracking                               │
│  • Model Checkpointing                                      │
└─────────────────────────────────────────────────────────────┘
```

### **Technical Innovations**

#### **1. K-mer Tokenization for DNA**

Adapted NLP tokenization techniques for genomic sequences:

```python
# 4-mer tokenization reduces vocabulary from 4^6 (4096) to 4^4 (256)
# while maintaining biological signal
sequence = "ATCGATCG"
kmers = ["ATCG", "TCGA", "CGAT", "GATC", "ATCG"]
```

**Benefits:**
- Smaller vocabulary → less overfitting on small data
- Preserves local sequence context
- Computationally efficient

#### **2. Biological Data Augmentation**

Three augmentation strategies maintaining biological validity:

1. **Reverse Complement**: Legitimate DNA strand representation
2. **Random Cropping (90-95%)**: Simulates partial sequencing
3. **Point Mutations (0.5-1%)**: Simulates natural variation

Expanded training data: 157 → 471 samples (3x increase)

#### **3. Small-Data Regularization**

- **High dropout (50%)**: Prevents memorization
- **Batch normalization**: Stabilizes training with small batches
- **Strong weight decay (0.05)**: L2 regularization
- **Early stopping (patience=20)**: Prevents overfitting

---

## 🛠️ Technical Stack

### **Core Technologies**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | PyTorch 2.0+ | Model training & inference |
| **Bioinformatics** | BioPython | Sequence manipulation |
| **Transformers** | Hugging Face | ESM-2 model loading |
| **Experiment Tracking** | MLflow | Hyperparameter logging |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Visualization** | Matplotlib, Seaborn | Results visualization |

### **Model Architectures Implemented**

1. **ESM-2 Fine-tuning** (35M parameters)
   - Pre-trained on 250M protein sequences
   - Frozen encoder + trainable classification head
   - Learning rate: 2e-5

2. **Custom CNN** (66K parameters)
   - Single Conv1D layer (128 filters, kernel=5)
   - Batch normalization
   - Adaptive max pooling
   - Two-layer classifier with dropout

3. **Simple CNN** (66K parameters, optimized for small data)
   - 4-mer tokenization (vocab=256)
   - Strong regularization (dropout=0.5)
   - Batch size=32 for stability

---

## 📁 Project Structure

```
viral-genome-classifier/
├── configs/                    # YAML configuration files
│   ├── base_config.yaml
│   ├── esm2_config.yaml
│   ├── cnn_config.yaml
│   └── working_config.yaml
├── data/
│   ├── raw/                   # Downloaded FASTA files
│   ├── processed/             # Cleaned sequences
│   └── splits/                # Train/val/test CSVs
├── src/
│   ├── data/
│   │   ├── download.py        # NCBI data collection
│   │   ├── preprocessing.py   # Quality control
│   │   ├── dataset.py         # PyTorch Dataset
│   │   ├── tokenization.py    # K-mer tokenizer
│   │   └── augmentation.py    # Data augmentation
│   ├── models/
│   │   ├── esm_classifier.py  # ESM-2 fine-tuning
│   │   ├── cnn_classifier.py  # Custom CNN
│   │   └── simple_cnn.py      # Simplified CNN
│   ├── training/
│   │   └── trainer.py         # Training loop
│   └── evaluation/
│       └── evaluator.py       # Metrics & visualization
├── scripts/
│   ├── prepare_dataset.py     # Data pipeline
│   ├── train.py              # ESM-2 training
│   ├── train_cnn.py          # CNN training
│   ├── train_working.py      # Best approach
│   └── evaluate.py           # Model evaluation
├── notebooks/                 # Jupyter analysis
├── models/checkpoints/        # Saved models
└── results/                   # Outputs
    ├── figures/
    ├── reports/
    └── predictions/
```

---

## 🚀 Quick Start

### **Installation**

```bash
# Clone repository
git clone https://github.com/shalakapadalkar16/viral-genome-classifier.git
cd viral-genome-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your NCBI credentials
```

### **Data Collection**

```bash
# Download viral genomes from NCBI
python scripts/prepare_dataset.py --config configs/base_config.yaml --download

# Preprocess and create splits
python scripts/prepare_dataset.py --config configs/base_config.yaml --process
```

### **Training**

```bash
# Train with data augmentation (recommended)
python scripts/train_working.py --config configs/working_config.yaml

# Monitor training with MLflow
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://localhost:5000
```

### **Evaluation**

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --config configs/working_config.yaml \
    --checkpoint models/checkpoints/best_model.pt
```

---

## 📈 Experiment Results

### **Model Comparison**

| Model | Parameters | Train Acc | Val Acc | Val F1 | Notes |
|-------|-----------|-----------|---------|--------|-------|
| ESM-2 (frozen) | 35M | 23% | 24% | 0.14 | Protein→DNA mismatch |
| CNN (6-mer) | 1.5M | 38% | 29% | 0.28 | Overfitting on small data |
| **Simple CNN (4-mer + aug)** | **66K** | **31%** | **32%** | **0.27** | **Best generalization** |

### **Data Augmentation Impact**

| Configuration | Train Samples | Val F1 | Observation |
|--------------|---------------|--------|-------------|
| No augmentation | 157 | 0.18 | Severe overfitting |
| 2x augmentation | 314 | 0.23 | Improved stability |
| **3x augmentation** | **471** | **0.27** | **Best performance** |
| 5x augmentation | 785 | 0.25 | Diminishing returns |

---

## 🔬 Key Learnings

### **1. Transfer Learning Limitations**

**Finding**: ESM-2 (protein language model) achieved only 24% accuracy despite 35M parameters.

**Analysis**: 
- Protein sequences use 20 amino acids; DNA uses 4 nucleotides
- ESM-2's attention patterns learned on protein structure don't transfer
- Vocabulary mismatch creates semantic gap

**Implication**: Domain-specific pre-training (like DNABERT) or training from scratch is necessary for genomic tasks.

### **2. Data Requirements for Genomics**

**Finding**: Performance plateaus around 32% with 225 total samples.

**Analysis**:
- Genomic diversity within viral families is high (mutation rates)
- 31 samples/class insufficient to capture variation
- Validation set (6-7 samples/class) too small for reliable metrics

**Implication**: Genomic classification requires 500-1000+ samples per class for clinical-grade accuracy.

### **3. Small-Data Strategies**

**What Worked**:
- ✅ Biological data augmentation (reverse complement)
- ✅ Strong regularization (dropout=0.5, weight decay=0.05)
- ✅ Smaller models (66K vs 35M parameters)
- ✅ K-mer tokenization (reduces vocabulary)

**What Didn't Work**:
- ❌ Large pre-trained models without domain match
- ❌ Complex architectures (3-layer CNNs overfit)
- ❌ Low regularization (dropout <0.3)

---

## 🎯 Production Roadmap

### **Phase 1: Data Scaling** (Current → 75% accuracy)

- [ ] Collect 500 sequences per family (2,500 total)
- [ ] Add more virus families (10+ families)
- [ ] Implement active learning for efficient labeling
- [ ] Balance dataset across families

### **Phase 2: Model Improvement** (75% → 85% accuracy)

- [ ] Fine-tune DNABERT (DNA-specific pre-training)
- [ ] Implement ensemble methods
- [ ] Add attention visualization for interpretability
- [ ] Optimize sequence length handling

### **Phase 3: Deployment** (85%+ accuracy)

- [ ] REST API for inference
- [ ] Docker containerization
- [ ] CI/CD pipeline with automated testing
- [ ] Model monitoring and drift detection
- [ ] Regulatory documentation (FDA 510(k) preparation)

---

## 📚 References & Related Work

### **Foundational Papers**

- **ESM-2**: Lin et al. (2022) "Language models of protein sequences at scale" - *Science*
- **DNABERT**: Ji et al. (2021) "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome" - *Bioinformatics*

### **Genomic Classification**

- Randhawa et al. (2019) "Machine learning using intrinsic genomic signatures for rapid classification of novel pathogens"
- Tampuu et al. (2019) "ViraMiner: Deep learning on raw DNA sequences for identifying viral genomes in human samples"

### **Small-Data Learning**

- Hernández-García & König (2018) "Data augmentation instead of explicit regularization"
- Shorten & Khoshgoftaar (2019) "A survey on Image Data Augmentation for Deep Learning"

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Data Collection**: Scripts for additional viral databases (GISAID, ViPR)
2. **Model Architectures**: Implement DNABERT, Nucleotide Transformer
3. **Evaluation**: Add sequence-level interpretability (attention maps)
4. **Documentation**: Tutorials for extending to new virus families

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Shalaka Padalkar**

- 🔗 [LinkedIn](https://www.linkedin.com/in/shalaka-padalkar/)
- 💻 [GitHub](https://github.com/shalakapadalkar16)
- 📧 Email: shalakapkar@gmail.com
- 🌐 [Portfolio](https://shalakapadalkar16.github.io/portfolio/)

---

## 🙏 Acknowledgments

- **NCBI GenBank** for viral genome sequences
- **Meta AI** for ESM-2 pre-trained models
- **Hugging Face** for transformer infrastructure
- **BioPython** community for sequence processing tools

---

**Built with 🧬 for advancing computational biology and diagnostic ML**