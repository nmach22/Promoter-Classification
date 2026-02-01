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