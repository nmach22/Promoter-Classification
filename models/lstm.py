import torch
import torch.nn as nn


"""
PromoterLSTM: Long Short-Term Memory Network for Promoter Classification

## Architecture Overview
This model processes DNA sequences using stacked LSTM layers followed by a 
fully-connected classifier head. It supports both one-hot encoding and k-mer 
embedding for sequence representation.

LSTMs use gating mechanisms (input, forget, and output gates) to better capture 
long-term dependencies in sequences compared to vanilla RNNs.

## DNA-Specific Improvements (NEW!)
This improved version includes several enhancements specifically for DNA sequences:
1. **CNN Layer for K-mer Pattern Extraction**: Convolutional layer before LSTM 
   captures local motifs and patterns (like TATA box, GC-rich regions)
2. **Layer Normalization**: Stabilizes training and improves convergence
3. **Multiple Pooling Strategies**: Choose between 'last', 'mean', 'max', or 
   'attention' to aggregate sequence information
4. **Flexible Architecture**: All improvements can be toggled on/off for experimentation

## Hyperparameter Guide

### `hidden_dim` (int, default=32)
**Controls the model's learning capacity**
- Determines the size of the LSTM's internal state vectors (both hidden and cell states)
- Larger values → More parameters → Higher capacity to learn complex patterns
- **Effect**: With bidirectional=True, the effective output dimension becomes `2 × hidden_dim`
- **Recommendations**:
  - Small sequences (< 200 bp): 32-64
  - Medium sequences (200-500 bp): 64-128
  - Large sequences (> 500 bp): 128-256
- **Note**: LSTMs can handle longer sequences better than RNNs due to their gating mechanism

### `num_layers` (int, default=1)
**Controls vertical depth of the LSTM**
- Creates a stack of LSTM layers where output of layer N feeds into layer N+1
- Deeper networks can capture more hierarchical patterns
- **Options**:
  - `1`: Single LSTM layer (shallow, fast training)
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
- Controls the fully-connected layers between LSTM output and final prediction
- Each value creates a layer with: Linear → ReLU → Dropout (if dropout > 0)
- **Options**:
  - `None` or `[]`: Direct LSTM → Output (single linear layer)
  - `[64]`: Adds one intermediate layer of size 64
  - `[128, 64]`: Two layers with decreasing size (funnel architecture)
  - `[256, 128, 64]`: Deep classifier (use with caution, may overfit)
- **Best Practice**: Start simple (None or [64]), add complexity if needed

### `dropout` (float, default=0.0)
**Regularization to prevent overfitting**
- Applied in two places:
  1. Between LSTM layers (when num_layers > 1)
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

### `use_cnn` (bool, default=True)
**Enables CNN layer for DNA motif detection**
- `True`: Adds Conv1D layer before LSTM to extract local k-mer patterns
- `False`: Direct embedding → LSTM (original architecture)
- **Benefits**: CNNs excel at detecting local motifs (e.g., TATA box, CAAT box)
- **Recommendation**: Keep `True` for DNA sequences, especially promoters

### `cnn_kernel_size` (int, default=7)
**Size of the convolutional kernel**
- Controls the length of DNA patterns the CNN can detect
- **Common values**:
  - `3-5`: Short motifs
  - `7-9`: Medium motifs (TATA box is ~6-8bp)
  - `11-15`: Longer regulatory elements
- **Recommendation**: 7 is a good default for most promoter motifs

### `cnn_out_channels` (int or None, default=None)
**Number of filters in CNN layer**
- `None`: Uses same as input dimension (conservative)
- Larger values: More diverse feature extraction
- **Recommendation**: Start with None, increase (e.g., 64, 128) if needed

### `use_layer_norm` (bool, default=True)
**Enables layer normalization for training stability**
- Normalizes activations to have mean=0, std=1
- **Benefits**: Faster convergence, less sensitive to initialization
- **Recommendation**: Keep `True` for better training stability

### `pooling_strategy` (str, default='last')
**How to aggregate LSTM outputs into a fixed-size vector**
- `'last'`: Use final hidden state (traditional LSTM)
- `'mean'`: Average all hidden states across sequence
- `'max'`: Take maximum activation across sequence
- `'attention'`: Learn to weight important positions (most sophisticated)
- **Recommendations**:
  - Start with `'last'` (default, fast)
  - Try `'attention'` for better performance (slightly slower)
  - `'mean'` and `'max'` are good middle ground

## LSTM vs RNN: Key Differences
- **Gating Mechanism**: LSTMs have input, forget, and output gates for better control
- **Memory**: LSTMs maintain both hidden state (h) and cell state (c)
- **Long-term Dependencies**: LSTMs handle longer sequences more effectively
- **Vanishing Gradients**: LSTMs are less susceptible to vanishing gradient problems
- **Performance**: Generally outperforms vanilla RNN, especially on longer sequences
- **Complexity**: More parameters per layer (~4x compared to RNN)

## Example Configurations

### Minimal (Fast, Baseline) - Without DNA-specific improvements
```python
model = PromoterLSTM(
    hidden_dim=32, 
    num_layers=1,
    use_cnn=False,
    use_layer_norm=False
)
```

### Standard (Recommended Starting Point) - With DNA improvements
```python
model = PromoterLSTM(
    hidden_dim=64,
    num_layers=2,
    bidirectional=True,
    fc_hidden_dims=[64],
    dropout=0.3,
    use_cnn=True,           # Enable CNN for motif detection
    cnn_kernel_size=7,      # Detect 7bp motifs
    use_layer_norm=True,    # Enable normalization
    pooling_strategy='mean' # Mean pooling over sequence
)
```

### Advanced (With Attention) - Best for complex patterns
```python
model = PromoterLSTM(
    hidden_dim=128,
    num_layers=2,
    bidirectional=True,
    fc_hidden_dims=[128, 64],
    dropout=0.3,
    use_cnn=True,
    cnn_kernel_size=9,
    cnn_out_channels=128,
    use_layer_norm=True,
    pooling_strategy='attention'  # Learn which positions are important
)
```

### Deep with K-mer Embeddings (Maximum Capacity)
```python
model = PromoterLSTM(
    vocab_size=256,         # For 4-mers
    embed_dim=16,
    hidden_dim=128,
    num_layers=3,
    bidirectional=True,
    fc_hidden_dims=[256, 128],
    dropout=0.4,
    use_cnn=True,
    cnn_kernel_size=7,
    use_layer_norm=True,
    pooling_strategy='attention'
)
```
"""

class PromoterLSTM(nn.Module):
    def __init__(
            self,
            vocab_size=None,
            embed_dim=None,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            fc_hidden_dims=None,
            use_cnn=True,
            cnn_kernel_size=7,
            cnn_out_channels=None,
            use_layer_norm=True,
            pooling_strategy='last'):
        """
        Args:
            vocab_size: Size of vocabulary for embedding (None for one-hot encoding)
            embed_dim: Embedding dimension (None for one-hot encoding)
            hidden_dim: Hidden dimension of LSTM layers
            num_layers: Number of stacked LSTM layers (default: 1)
            dropout: Dropout probability between LSTM layers (default: 0.0)
            bidirectional: Whether to use bidirectional LSTM (default: False)
            fc_hidden_dims: List of hidden dimensions for classifier head (e.g., [64, 32])
                           If None, uses a single linear layer
            use_cnn: Whether to use CNN layer for k-mer feature extraction (default: True)
            cnn_kernel_size: Kernel size for CNN layer (default: 7, good for motifs)
            cnn_out_channels: Output channels for CNN (default: None, uses lstm_input_size)
            use_layer_norm: Whether to use layer normalization (default: True)
            pooling_strategy: How to aggregate LSTM outputs ('last', 'mean', 'max', 'attention')
        """
        super().__init__()

        # Convert None to empty list
        fc_hidden_dims = fc_hidden_dims or []
        self.pooling_strategy = pooling_strategy
        self.use_cnn = use_cnn

        # If vocab_size is provided, use Embedding (K-mer mode)
        if vocab_size:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            lstm_input_size = embed_dim
        else:
            # No embedding (One-hot mode)
            self.embedding = nn.Identity()
            lstm_input_size = 4

        # Optional CNN layer for k-mer pattern extraction (DNA-specific improvement)
        if use_cnn:
            cnn_out_channels = cnn_out_channels or lstm_input_size
            self.cnn = nn.Sequential(
                nn.Conv1d(
                    in_channels=lstm_input_size,
                    out_channels=cnn_out_channels,
                    kernel_size=cnn_kernel_size,
                    padding=cnn_kernel_size // 2
                ),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            lstm_input_size = cnn_out_channels
        else:
            self.cnn = None

        # Optional Layer Normalization (improves training stability)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_input_size)
        else:
            self.layer_norm = None

        # Deeper LSTM with multiple layers
        self.lstm = nn.LSTM(
            lstm_input_size,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Calculate the output dimension from LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention mechanism for pooling (optional)
        if pooling_strategy == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_dim // 2, 1)
            )
        else:
            self.attention = None

        # Deeper classifier head
        layers = []
        input_dim = lstm_output_dim

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
        x = self.embedding(x)  # (batch, seq_len, input_dim)

        # Apply CNN for k-mer feature extraction (if enabled)
        if self.use_cnn:
            # Conv1d expects (batch, channels, seq_len)
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            x = x.permute(0, 2, 1)  # Back to (batch, seq_len, channels)

        # Apply layer normalization (if enabled)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # LSTM returns (output, (hidden, cell))
        # output: (batch, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_dim)
        lstm_output, (hidden, cell) = self.lstm(x)

        # Apply pooling strategy
        if self.pooling_strategy == 'last':
            # Use last hidden state (traditional approach)
            if self.lstm.bidirectional:
                # Concatenate forward and backward hidden states from last layer
                pooled = hidden[-2:].transpose(0, 1).contiguous().view(x.size(0), -1)
            else:
                pooled = hidden[-1]

        elif self.pooling_strategy == 'mean':
            # Mean pooling over sequence
            pooled = lstm_output.mean(dim=1)

        elif self.pooling_strategy == 'max':
            # Max pooling over sequence
            pooled = lstm_output.max(dim=1)[0]

        elif self.pooling_strategy == 'attention':
            # Attention-based pooling
            # Compute attention weights
            attn_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            # Weighted sum
            pooled = (lstm_output * attn_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Classification head
        output = self.fc(pooled)

        return output.squeeze(-1)

#
# model = PromoterLSTM(
#     hidden_dim=64,
#     num_layers=2,
#     bidirectional=True,
#     fc_hidden_dims=[64],
#     dropout=0.3,
#     use_cnn=True,           # Enable CNN
#     cnn_kernel_size=7,      # 7bp motifs
#     use_layer_norm=True,    # Stabilize training
#     pooling_strategy='attention'  # Learn importance
# )