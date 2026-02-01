import torch
from torch.utils.data import Dataset
import numpy as np
from Bio import SeqIO
from .encoding_functions import *

class FastaDataset(Dataset):
    def __init__(self, pos_fasta, neg_fasta, encoding_func = one_hot_encode,seq_len=300):
        self.seq_len = seq_len
        self.sequences = []
        self.labels = []
        self.encoding_func = encoding_func

        self._load_fasta(pos_fasta, label=1)
        self._load_fasta(neg_fasta, label=0)

    def _load_fasta(self, fasta_path, label):
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq).upper()
            self.sequences.append(seq)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    # def one_hot_encode(self, seq):
    #     mapping = {
    #         "A": [1,0,0,0],
    #         "C": [0,1,0,0],
    #         "G": [0,0,1,0],
    #         "T": [0,0,0,1]
    #     }

    #     encoded = np.zeros((self.seq_len, 4), dtype=np.float32)

    #     for i in range(min(len(seq), self.seq_len)):
    #         encoded[i] = mapping.get(seq[i], [0,0,0,0])

    #     return encoded

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        x = self.encoding_func(seq,self.seq_len)
        x = torch.from_numpy(x)
        y = torch.tensor(label, dtype=torch.long)

        return x, y
