import torch.nn as nn

class Vex(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.GELU(),

            nn.Conv2d(8, 12, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.GELU(),

            nn.Conv2d(12, 16, 3, 1),
            nn.GELU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.linear_layer(y_conv)
        return y