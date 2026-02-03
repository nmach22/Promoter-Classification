import torch.nn as nn


"""
## Parameter Effects

### `num_layers` - Makes RNN deeper vertically
- `num_layers=1`: Single RNN layer (shallow)
- `num_layers=2`: 2 stacked RNN layers
- `num_layers=3`: 3 stacked RNN layers (deep)
- **Tip**: Values 2-4 are common. Higher values may need more data.

### `bidirectional` - Processes sequences in both directions
- `False`: Reads sequence left-to-right only
- `True`: Reads both left-to-right and right-to-left, **doubles output dimension**
- **Tip**: Usually improves performance for DNA sequences

### `fc_hidden_dims` - Makes classifier head deeper
- `None` or `[]`: Direct connection (RNN → Output)
- `[64]`: One hidden layer
- `[128, 64]`: Two hidden layers  
- `[256, 128, 64]`: Three hidden layers (very deep)
- **Tip**: Each layer adds: Linear → ReLU → Dropout

### `dropout` - Regularization
- Applied between RNN layers (if `num_layers > 1`)
- Applied after each FC hidden layer
- **Tip**: 0.2-0.5 for deep models, 0.0 for shallow

### `hidden_dim` - Controls capacity
- Larger values = more parameters = more capacity
- With `bidirectional=True`, output is `2 * hidden_dim`
"""

class PromoterRNN(nn.Module):
    def __init__(
            self,
            vocab_size=None,
            embed_dim=None,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            fc_hidden_dims=None):
        """
        Args:
            vocab_size: Size of vocabulary for embedding (None for one-hot encoding)
            embed_dim: Embedding dimension (None for one-hot encoding)
            hidden_dim: Hidden dimension of RNN layers
            num_layers: Number of stacked RNN layers (default: 1)
            dropout: Dropout probability between RNN layers (default: 0.0)
            bidirectional: Whether to use bidirectional RNN (default: False)
            fc_hidden_dims: List of hidden dimensions for classifier head (e.g., [64, 32])
                           If None, uses a single linear layer
        """
        super().__init__()

        # Convert None to empty list
        fc_hidden_dims = fc_hidden_dims or []

        # If vocab_size is provided, use Embedding (K-mer mode)
        if vocab_size:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            rnn_input_size = embed_dim
        else:
            # No embedding (One-hot mode)
            self.embedding = nn.Identity()
            rnn_input_size = 4

        # Deeper RNN with multiple layers
        self.rnn = nn.RNN(
            rnn_input_size,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Calculate the output dimension from RNN
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Deeper classifier head

        layers = []
        input_dim = rnn_output_dim

        # Add hidden layers
        for hidden in fc_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden

        # Add final output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # x is (batch, seq_len) for k-mer OR (batch, seq_len, 4) for one-hot
        x = self.embedding(x)
        _, hidden = self.rnn(x)

        # For bidirectional RNN, concatenate forward and backward hidden states
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        if self.rnn.bidirectional:
            # Take the last layer's forward and backward hidden states
            hidden = hidden[-2:].transpose(0, 1).contiguous().view(x.size(0), -1)
        else:
            # Take the last layer's hidden state
            hidden = hidden[-1]

        x = self.fc(hidden)

        return x.squeeze(-1)