import itertools
import numpy as np


def one_hot_encode(seq, max_seq_len):
    mapping = {
        "A": [1,0,0,0],
        "C": [0,1,0,0],
        "G": [0,0,1,0],
        "T": [0,0,0,1]
    }

    encoded = np.zeros((max_seq_len, 4), dtype=np.float32)

    for i in range(min(len(seq), max_seq_len)):
        encoded[i] = mapping.get(seq[i], [0,0,0,0])

    return encoded


def token_encode(seq, seq_len):
    vocab = {"A":0, "C":1, "G":2, "T":3}
    encoded = np.zeros(seq_len, dtype=np.int64)

    for i in range(min(len(seq), seq_len)):
        encoded[i] = vocab.get(seq[i], 0)

    return encoded

class KmerEncoding:
    def __init__(self, k=3):
        self.k = k
        self.bases = ['A', 'C', 'G', 'T']
        self.kmers = [''.join(p) for p in itertools.product(self.bases, repeat=k)]
        self.vocab = {kmer: i + 1 for i, kmer in enumerate(self.kmers)}
        self.vocab['<PAD>'] = 0

    def __call__(self, seq, max_seq_len):
        # max_seq_len here refers to the number of k-mers we want
        indices = np.zeros(max_seq_len, dtype=np.int64)

        # Extract k-mers
        for i in range(min(len(seq) - self.k + 1, max_seq_len)):
            kmer = seq[i:i + self.k]
            indices[i] = self.vocab.get(kmer, 0)

        return indices
