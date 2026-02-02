import torch.nn as nn

class PromoterRNN(nn.Module):
    def __init__(self, vocab_size=None, embed_dim=None, hidden_dim=32):
        super().__init__()

        # If vocab_size is provided, use Embedding (K-mer mode)
        if vocab_size:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            rnn_input_size = embed_dim
        else:
            # No embedding (One-hot mode)
            self.embedding = nn.Identity()
            rnn_input_size = 4

        self.rnn = nn.RNN(rnn_input_size, hidden_dim, batch_first=True)
        self.fc = nn.Sigmoid(hidden_dim, 1)

    def forward(self, x):
        # x is (batch, seq_len) for k-mer OR (batch, seq_len, 4) for one-hot
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        return self.fc(hidden.squeeze(0))