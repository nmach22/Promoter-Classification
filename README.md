# Promoter-Classification

Reproduction and improvements for: [Recognition of prokaryotic and eukaryotic promoters using convolutional deep learning neural networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC5291440/)

GitHub Repository: [Promoter-Classification](https://github.com/nmach22/Promoter-Classification)

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

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Usage

See the notebooks in `notebooks/` for examples of training and evaluating models.
