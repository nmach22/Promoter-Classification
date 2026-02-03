"""
Examples of how to instantiate the deeper PromoterRNN model with various configurations.
"""
from models.rnn import PromoterRNN

# ============================================================================
# Example 1: Simple shallow model (original behavior)
# ============================================================================
model_simple = PromoterRNN(
    vocab_size=65,
    embed_dim=16,
    hidden_dim=32,
    num_layers=1,         # Single RNN layer
    dropout=0.0,
    bidirectional=False,
    fc_hidden_dims=None   # Direct connection: RNN -> output
)
# Architecture: Input(4) -> RNN(32) -> Linear(32, 1) -> Sigmoid


# ============================================================================
# Example 2: Deep RNN with multiple stacked layers
# ============================================================================
model_deep_rnn = PromoterRNN(
    vocab_size=65,
    embed_dim=16,
    hidden_dim=64,
    num_layers=3,         # 3 stacked RNN layers
    dropout=0.3,          # Dropout between RNN layers
    bidirectional=False,
    fc_hidden_dims=None
)
# Architecture: Input(4) -> RNN(64, 3 layers) -> Linear(64, 1) -> Sigmoid


# ============================================================================
# Example 3: Bidirectional RNN for better context
# ============================================================================
model_bidirectional = PromoterRNN(
    vocab_size=65,
    embed_dim=16,
    hidden_dim=32,
    num_layers=2,
    dropout=0.2,
    bidirectional=True,   # Bidirectional RNN
    fc_hidden_dims=None
)
# Architecture: Input(4) -> BiRNN(32, 2 layers) -> Linear(64, 1) -> Sigmoid
# Note: Output is 64 because bidirectional doubles the hidden dimension


# ============================================================================
# Example 4: Deep fully connected classifier head
# ============================================================================
model_deep_fc = PromoterRNN(
    vocab_size=65,
    embed_dim=16,
    hidden_dim=32,
    num_layers=1,
    dropout=0.0,
    bidirectional=False,
    fc_hidden_dims=[64, 32]  # Two hidden layers in classifier
)
# Architecture: Input(4) -> RNN(32) -> Linear(32, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> Sigmoid


# ============================================================================
# Example 5: Very deep model with everything
# ============================================================================
model_very_deep = PromoterRNN(
    vocab_size=65,
    embed_dim=16,
    hidden_dim=128,
    num_layers=4,              # 4 stacked RNN layers
    dropout=0.3,               # Dropout in both RNN and FC
    bidirectional=True,        # Bidirectional
    fc_hidden_dims=[256, 128, 64]  # 3 hidden layers in classifier
)
# Architecture: Input(4) -> BiRNN(128, 4 layers) -> Linear(256, 256) -> ReLU -> Dropout
#               -> Linear(256, 128) -> ReLU -> Dropout -> Linear(128, 64) -> ReLU -> Dropout
#               -> Linear(64, 1) -> Sigmoid


# ============================================================================
# Example 6: K-mer mode with embeddings
# ============================================================================
model_kmer = PromoterRNN(
    vocab_size=256,        # 4^4 = 256 possible 4-mers
    embed_dim=32,          # Embedding dimension
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    bidirectional=True,
    fc_hidden_dims=[128, 64]
)
# Architecture: Input(indices) -> Embedding(256, 32) -> BiRNN(64, 2 layers)
#               -> Linear(128, 128) -> ReLU -> Dropout -> Linear(128, 64) -> ReLU -> Dropout
#               -> Linear(64, 1) -> Sigmoid


# ============================================================================
# Example 7: Minimal dropout, focus on depth
# ============================================================================
model_deep_no_dropout = PromoterRNN(
    vocab_size=65,
    embed_dim=16,
    hidden_dim=64,
    num_layers=5,          # Very deep RNN
    dropout=0.0,           # No dropout (not recommended for deep models)
    bidirectional=False,
    fc_hidden_dims=[128, 64, 32]  # Deep FC head
)


# ============================================================================
# Example 8: Recommended configuration for promoter classification
# ============================================================================
model_recommended = PromoterRNN(
    vocab_size=65,
    embed_dim=16,
    hidden_dim=64,
    num_layers=2,          # 2 layers is often a good balance
    dropout=0.3,           # Moderate dropout for regularization
    bidirectional=True,    # Bidirectional captures context better
    fc_hidden_dims=[128]   # One hidden layer in classifier
)
# Architecture: Input(4) -> BiRNN(64, 2 layers) -> Linear(128, 128) -> ReLU -> Dropout
#               -> Linear(128, 1) -> Sigmoid
