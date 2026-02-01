import numpy as np

from utils.encoding_functions import token_encode
import os
import yaml

from pathlib import Path

from utils.fasta_dataset import FastaDataset

_PATH_TO_ROOT = Path.cwd().parent.absolute()
_DEFAULT_CONFIG_PATH = os.path.join(_PATH_TO_ROOT, 'config', 'config.yaml')


def splitdata(full_dataset):
    # 1. Get indices and targets (assumes your dataset has a .targets or similar attribute)
    indices = np.arange(len(full_dataset))
    targets = full_dataset.labels

    # 2. Split indices using scikit-learn for stratification
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=42
    )

    # 3. Create Subset objects
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)


def main():
    with open(_DEFAULT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    fasta_positive = config['data']['bacillus']['promoter_fasta']
    fasta_negative = config['data']['bacillus']['promoter_fasta']
    seq_length = config['data']['bacillus']['seq_len']

    full_pos_path = os.path.join(_PATH_TO_ROOT, fasta_positive)
    full_neg_path = os.path.join(_PATH_TO_ROOT, fasta_negative)

    data_positive = FastaDataset(full_pos_path, label=1, encoding_func=token_encode, seq_len=seq_length)
    data_negative = FastaDataset(full_neg_path, label=0, encoding_func=token_encode, seq_len=seq_length)


    splitdata(data_positive)


if __name__ == "__main__":
    main()