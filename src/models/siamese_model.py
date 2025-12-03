import torch
import torch.nn as nn
from src.config import DROPOUT

class SiameseNetwork(nn.Module):
    """Siamese Network architecture for face identification."""
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveMaxPool2d((6, 6))

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.Sigmoid(),
            nn.Dropout(p=DROPOUT)
        )

        self.out = nn.Linear(4096, 1)

    def forward_once(self, x) -> torch.Tensor:
        """
        Forward pass for one branch of the Siamese Network.
        Args:
            x (Tensor): Input image tensor.
        Returns:
            Tensor: Output feature vector.
        """
        output = self.cnn(x)

        # Pass through Adaptive Pool (Reduces 24x24 -> 6x6)
        output = self.adaptive_pool(output)

        output = output.view(output.size()[0], -1)

        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        l1_distance = torch.abs(output1 - output2)
        score = torch.sigmoid(self.out(l1_distance))
        return score
