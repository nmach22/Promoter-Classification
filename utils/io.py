from typing import List
from Bio import SeqIO


def load_fasta(path: str):
    """Yield SeqRecord objects from a FASTA file."""
    for rec in SeqIO.parse(path, "fasta"):
        yield rec
