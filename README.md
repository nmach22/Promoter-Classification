# Promoter-Classification

Reproduction and improvements for: https://pmc.ncbi.nlm.nih.gov/articles/PMC5291440/

## Project Structure

```
Promoter-Classification/
├── config/                    # Configuration files for experiments
│   └── __init__.py
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
│   ├── Mouse_tata_dbtss.fa
│   └── Mouse_tata.fa
│
├── eval/                      # Evaluation scripts and metrics
│   └── __init__.py
│
├── losses/                    # Custom loss functions
│   ├── __init__.py
│   ├── cross_entropy.py       # Cross entropy loss implementation
│   └── focal_loss.py          # Focal loss implementation
│
├── models/                    # Model architectures for promoter classification
│   ├── __init__.py
│   └── simple_cnn.py          # Simple CNN model
│
├── notebooks/                 # Jupyter notebooks for analysis and training
│   ├── __init__.py
│   ├── colab_instructions.md  # Instructions for running on Google Colab
│   └── train_cnn.ipynb        # Model training notebook
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── io.py                  # I/O utilities for data processing
│
├── requirements.txt           # Python dependencies
├── LICENSE                    # Project license
└── README.md                  # This file
```

## Overview

This project implements deep learning models for promoter sequence classification, with support for multiple organisms including:
- *Arabidopsis thaliana*
- *Bacillus*
- *E. coli*
- Human
- Mouse

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Usage

See the notebooks in `notebooks/` for examples of training and evaluating models.
