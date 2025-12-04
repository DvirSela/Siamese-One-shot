import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        """
        Siamese Network with interchangeable backbone.
        Args:
            backbone_name (str): 'resnet18', 'resnet34', 'resnet50', etc.
            pretrained (bool): Whether to load ImageNet weights.
        """
        super(SiameseNetwork, self).__init__()
        
        # 1. Load the Backbone
        # We use the factory pattern to make swapping easy
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.fc.in_features # 512
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.fc.in_features # 512
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.fc.in_features # 2048
        else:
            raise ValueError(f"Backbone {backbone_name} not supported yet.")

        # 2. Modify First Layer for Grayscale
        # ResNet expects 3 channels (RGB). We have 1 (Grayscale).
        # We replace conv1 with a 1-channel version.
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Initialize the new 1-channel weights by averaging the original 3-channel weights
        # This preserves the pre-trained filters' spatial structure.
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        # 3. Remove the Classification Head (fc)
        # We replace the final 'fc' layer with a simple Identity, 
        # so we get the raw feature vector from the Global Average Pooling layer.
        self.backbone.fc = nn.Identity()

        # 4. Projection / Embedding Head
        # Projects high-dim features to your desired embedding size
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
            # No Dropout needed here for ResNet usually, BN handles it
        )

    def forward_once(self, x):
        # Pass through ResNet backbone
        # Output shape: (Batch, in_features) e.g., (Batch, 512)
        features = self.backbone(x)
        
        # Pass through projection head
        embeddings = self.fc(features)
        
        return embeddings

    def forward(self, input1, input2):
        v1 = self.forward_once(input1)
        v2 = self.forward_once(input2)
        return v1, v2