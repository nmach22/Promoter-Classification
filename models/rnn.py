import torch.nn as nn


"""
PromoterRNN: Recurrent Neural Network for Promoter Classification

## Architecture Overview
This model processes DNA sequences using stacked RNN layers followed by a 
fully-connected classifier head. It supports both one-hot encoding and k-mer 
embedding for sequence representation.

## Hyperparameter Guide

### `hidden_dim` (int, default=32)
**Controls the model's learning capacity**
- Determines the size of the RNN's internal state vector
- Larger values → More parameters → Higher capacity to learn complex patterns
- **Effect**: With bidirectional=True, the effective output dimension becomes `2 × hidden_dim`
- **Recommendations**:
  - Small sequences (< 200 bp): 32-64
  - Medium sequences (200-500 bp): 64-128
  - Large sequences (> 500 bp): 128-256

### `num_layers` (int, default=1)
**Controls vertical depth of the RNN**
- Creates a stack of RNN layers where output of layer N feeds into layer N+1
- Deeper networks can capture more hierarchical patterns
- **Options**:
  - `1`: Single RNN layer (shallow, fast training)
  - `2-3`: Moderate depth (good balance, most common)
  - `4+`: Deep architecture (requires more data, prone to overfitting)
- **Note**: Dropout is automatically applied between layers when num_layers > 1

### `bidirectional` (bool, default=False)
**Enables processing sequences in both directions**
- `False`: Processes sequence left-to-right only (forward pass)
- `True`: Processes both left-to-right AND right-to-left, then concatenates results
- **Impact**: Doubles the output dimension (important for fc_hidden_dims sizing)
- **Benefits**: Captures context from both directions, often improves accuracy
- **Recommendation**: Usually set to `True` for DNA sequences since biological 
  signals can appear in either direction

### `fc_hidden_dims` (list or None, default=None)
**Defines the classifier head architecture**
- Controls the fully-connected layers between RNN output and final prediction
- Each value creates a layer with: Linear → ReLU → Dropout (if dropout > 0)
- **Options**:
  - `None` or `[]`: Direct RNN → Output (single linear layer)
  - `[64]`: Adds one intermediate layer of size 64
  - `[128, 64]`: Two layers with decreasing size (funnel architecture)
  - `[256, 128, 64]`: Deep classifier (use with caution, may overfit)
- **Best Practice**: Start simple (None or [64]), add complexity if needed

### `dropout` (float, default=0.0)
**Regularization to prevent overfitting**
- Applied in two places:
  1. Between RNN layers (when num_layers > 1)
  2. After each hidden layer in the classifier head
- Value represents probability of dropping a neuron during training
- **Range**: 0.0 (no dropout) to 0.5 (aggressive regularization)
- **Guidelines**:
  - `0.0`: Small datasets or shallow models
  - `0.2-0.3`: Standard choice for moderate depth
  - `0.4-0.5`: Deep models or signs of overfitting
- **Note**: Only active during training, automatically disabled during evaluation

### `vocab_size` & `embed_dim` (int or None, default=None)
**Controls input representation mode**
- **Mode 1 - One-hot encoding** (when both are None):
  - Input: `(batch, seq_len, 4)` for A, C, G, T
  - No embedding layer, direct processing
  - Good for: Small datasets, interpretability
- **Mode 2 - K-mer embeddings** (when both are provided):
  - Input: `(batch, seq_len)` with integer k-mer IDs
  - Learns dense representations for k-mer patterns
  - `vocab_size`: Total number of unique k-mers (e.g., 256 for 4-mers)
  - `embed_dim`: Size of learned embedding vector (typical: 8-32)
  - Good for: Larger datasets, capturing k-mer patterns

## Example Configurations

### Minimal (Fast, Baseline)
```python
model = PromoterRNN(hidden_dim=32, num_layers=1)
```

### Standard (Recommended Starting Point)
```python
model = PromoterRNN(
    hidden_dim=64,
    num_layers=2,
    bidirectional=True,
    fc_hidden_dims=[64],
    dropout=0.3
)
```

### Deep (Maximum Capacity)
```python
model = PromoterRNN(
    vocab_size=256,
    embed_dim=16,
    hidden_dim=128,
    num_layers=3,
    bidirectional=True,
    fc_hidden_dims=[256, 128],
    dropout=0.4
)
```
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