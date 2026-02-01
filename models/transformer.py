import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             (-math.log(5000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class DNATransformer(nn.Module):
    def __init__(
        self,
        seq_len=300,
        d_model=256,
        nhead=8,
        num_layers=6,
        k=7,            # k-mer size (conv kernel)
        num_classes=1,
        dropout=0.1
    ):
        super().__init__()

        # 4 nucleotides: A,C,G,T
        self.embedding = nn.Embedding(4, d_model)

        # Conv1D for k-mer learning
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=k,
            padding=k // 2
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # classifier
        self.fc = nn.Sequential(
          nn.Linear(d_model, num_classes),
          nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, L) integer tokens 0..3
        """

        # nucleotide embedding
        x = self.embedding(x)              # (B, L, D)

        # Conv expects (B, D, L)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)             # (B, L, D)

        # positional encoding
        x = self.pos_encoder(x)

        # transformer
        x = self.transformer(x)

        # global average pooling
        x = x.mean(dim=1)

        # classifier
        x = self.fc(x)

        return x.squeeze(-1)

