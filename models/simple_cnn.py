import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x expected shape: (batch, seq_len) ints 0-4 -> one-hot convert outside
        return self.net(x)

if __name__ == '__main__':
    m = SimpleCNN()
    x = torch.randn(4,5,200)
    print(m(x).shape)
