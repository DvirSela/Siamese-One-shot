import torch
import torch.nn as nn
class SimpleSiameseNetwork(nn.Module):
    """
    A simple CNN-based Siamese Network for sanity checking and overfitting tests.
    """
    def __init__(self):
        super(SimpleSiameseNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
        )

        # Final size: 8 × 15 × 15 = 1800
        self.fc_embed = nn.Sequential(
            nn.Linear(8 * 15 * 15, 64),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Linear(64, 1)

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        emb = self.fc_embed(x)
        return emb

    def forward(self, input1, input2, return_embeddings=False):
        emb1 = self.forward_once(input1)
        emb2 = self.forward_once(input2)

        l1_distance = torch.abs(emb1 - emb2)
        score = torch.sigmoid(self.out(l1_distance))

        if return_embeddings:
            return score, emb1, emb2

        return score