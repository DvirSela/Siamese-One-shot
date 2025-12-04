import torch
import torch.nn as nn
from src.config import DROPOUT

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Siamese Network for Contrastive Learning.
    Outputs raw feature embeddings instead of probability scores.
    """
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
            nn.ReLU(inplace=True)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = self.adaptive_pool(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2