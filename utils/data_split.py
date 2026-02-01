import numpy as np
from torch.utils.data import Subset

def dataset_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert train_ratio + val_ratio + test_ratio == 1.0

    labels = np.array(dataset.labels)

    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def split_indices(idxs):
        n = len(idxs)
        n_train = int(train_ratio * n)
        n_val   = int(val_ratio * n)
        train = idxs[:n_train]
        val   = idxs[n_train:n_train+n_val]
        test  = idxs[n_train+n_val:]
        return train, val, test

    pos_train, pos_val, pos_test = split_indices(idx_pos)
    neg_train, neg_val, neg_test = split_indices(idx_neg)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx   = np.concatenate([pos_val, neg_val])
    test_idx  = np.concatenate([pos_test, neg_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx)
    )
