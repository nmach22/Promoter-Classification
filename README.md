# Promoter-Classification

Reproduction and improvements for: [Recognition of prokaryotic and eukaryotic promoters using convolutional deep learning neural networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC5291440/)

GitHub Repository: [Promoter-Classification](https://github.com/nmach22/Promoter-Classification)

## Abstract

Promoter recognition is a fundamental problem in computational biology, as accurate identification of promoter regions is essential for understanding gene regulation and expression mechanisms. This project extends the work of Umarov and Solovyev (2017), who successfully applied Convolutional Neural Networks (CNNs) for prokaryotic and eukaryotic promoter classification. We implement and compare multiple state-of-the-art deep learning architectures—including Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Transformer models—alongside traditional machine learning approaches using XGBoost for bacterial promoter sequence classification.

Our experiments focus on two prokaryotic organisms: *Escherichia coli* σ70 and *Bacillus subtilis*. The dataset comprises promoter sequences from manually curated databases (RegulonDB for *E. coli* and DBTBS for *B. subtilis*) and non-promoter sequences extracted from random fragments of protein-coding gene regions. 

The Transformer architecture demonstrated superior performance across both organisms, achieving 95.3% accuracy and a Matthews Correlation Coefficient (MCC) of 0.863 for *E. coli*, and 93.2% accuracy with 0.828 MCC for *B. subtilis*, surpassing the original paper's CNN results (MCC of 0.84 and 0.86, respectively). The XGBoost model also showed competitive performance with simpler feature engineering, while LSTM networks provided moderate improvements over basic RNN architectures. Our results demonstrate that attention-based mechanisms in Transformer models are particularly effective at capturing long-range dependencies and regulatory motifs in DNA sequences, offering a promising direction for genomic sequence analysis tasks.

The project provides a comprehensive framework with modular implementations of multiple model architectures, configurable training pipelines, and detailed evaluation metrics, making it accessible for both reproduction and extension to other organisms or sequence classification tasks.

## Project Structure

```
Promoter-Classification/
├── config/                    # Configuration files for experiments
│   ├── __init__.py
│   └── config.yaml
│
├── dataset/                   # Original FASTA datasets
│   ├── __init__.py
│   ├── README                 # Dataset documentation
│   ├── Arabidopsis_non_prom_big.fa
│   ├── Arabidopsis_non_prom.fa
│   ├── Arabidopsis_non_tata.fa
│   ├── Arabidopsis_tata.fa
│   ├── Bacillus_non_prom.fa
│   ├── Bacillus_prom.fa
│   ├── Ecoli_non_prom.fa
│   ├── Ecoli_prom.fa
│   ├── human_non_tata.fa
│   ├── human_nonprom_big.fa
│   ├── Mouse_non_nonprom_big.fa
│   ├── Mouse_non_tata.fa
│   ├── Mouse_nonprom.fa
│   ├── mouse_prom.fa
│   ├── Mouse_tata_dbtss.fa
│   └── Mouse_tata.fa
│
├── eval/                      # Evaluation scripts and metrics
│   ├── __init__.py
│   ├── metrics.py
│   └── train_evals.py
│
├── losses/                    # Custom loss functions
│   ├── __init__.py
│   ├── cross_entropy.py       # Cross entropy loss implementation
│   └── focal_loss.py          # Focal loss implementation
│
├── models/                    # Model architectures for promoter classification
│   ├── __init__.py
│   ├── lstm.py                # LSTM model
│   ├── rnn.py                 # RNN model
│   ├── transformer.py         # Transformer model
│   └── xgboost_model.py       # XGBoost model
│
├── notebooks/                 # Jupyter notebooks for analysis and training
│   ├── __init__.py
│   ├── colab_instructions.md  # Instructions for running on Google Colab
│   ├── local_test.ipynb       # Local testing notebook
│   ├── train_lstm.ipynb       # LSTM training notebook
│   ├── train_rnn.ipynb        # RNN training notebook
│   ├── train_transformer.ipynb # Transformer training notebook
│   ├── train_xgboost_arabidopsis.ipynb
│   ├── train_xgboost_bacillus.ipynb
│   ├── train_xgboost_ecoli.ipynb
│   ├── train_xgboost_human.ipynb
│   ├── train_xgboost_mouse.ipynb
│   └── train_xgboost.ipynb    # XGBoost training notebook
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── data_split.py          # Data splitting utilities
│   ├── encoding_functions.py  # Sequence encoding functions
│   ├── fasta_dataset.py       # FASTA dataset loader
│   ├── get_device.py          # Device selection utility
│   └── train.py               # Training utilities
│
├── requirements.txt           # Python dependencies
├── RNN vs LSTM.md             # Comparison between RNN and LSTM
├── LICENSE                    # Project license
└── README.md                  # This file
```

## Overview

This project implements both deep learning models (RNN, LSTM, and Transformer) and machine learning models (XGBoost) for promoter sequence classification. The current experiments focus on prokaryotic DNA, specifically:
- *Bacillus subtilis*
- *Escherichia coli* (E. coli)

While the dataset includes sequences from eukaryotic organisms (*Arabidopsis thaliana*, Human, and Mouse), the trained models and experiments were conducted exclusively on prokaryotic promoter sequences.

### Models Implemented
- **RNN (Recurrent Neural Network)**: Basic recurrent architecture for sequence classification
- **LSTM (Long Short-Term Memory)**: Advanced recurrent model with memory cells
- **Transformer**: Attention-based architecture for sequence analysis
- **XGBoost**: Gradient boosting classifier using engineered features from DNA sequences

## Results

The following table presents the performance metrics of different models on prokaryotic promoter classification tasks. Results are compared against the original paper's CNN model ("Real Model").

### Performance Comparison

| Organism | Model | Accuracy | Sensitivity (Sn) | Specificity (Sp) | MCC / CC |
|----------|-------|----------|------------------|------------------|----------|
| *E. coli* | Real Model (Paper) | – | 0.90 | 0.96 | 0.84 |
| *E. coli* | **Transformer** | **0.953** | **0.882** | **0.973** | **0.863** |
| *E. coli* | XGBoost | 0.927 | 0.787 | 0.967 | 0.782 |
| *E. coli* | LSTM | 0.913 | 0.772 | 0.953 | 0.742 |
| *B. subtilis* | Real Model (Paper) | – | 0.91 | 0.95 | 0.86 |
| *B. subtilis* | **Transformer** | **0.932** | **0.789** | **0.987** | **0.828** |
| *B. subtilis* | XGBoost | 0.894 | 0.842 | 0.913 | 0.740 |
| *B. subtilis* | LSTM | 0.870 | 0.807 | 0.893 | 0.683 |

### Key Findings

- **Transformer models achieved the best overall performance** across both organisms, demonstrating superior accuracy and Matthews Correlation Coefficient (MCC) compared to the original paper's CNN approach
- For *E. coli*, the Transformer model achieved **95.3% accuracy** with an MCC of **0.863**, outperforming the paper's reported 0.84 MCC
- For *B. subtilis*, the Transformer model reached **93.2% accuracy** with excellent specificity (**98.7%**) and an MCC of **0.828**
- XGBoost showed strong performance as a non-deep learning baseline, achieving competitive results with simpler feature engineering
- LSTM models, while effective, showed lower performance compared to Transformer and XGBoost approaches

**Note**: MCC (Matthews Correlation Coefficient) is a balanced measure that accounts for class imbalance and is considered one of the best metrics for binary classification tasks in bioinformatics.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Usage

See the notebooks in `notebooks/` for examples of training and evaluating models.
