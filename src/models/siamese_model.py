import torch
import torch.nn as nn
from torch.nn import functional as F
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
        self.backbone_name = backbone_name

        # CNN-based 
        if backbone_name in ['resnet18', 'resnet34', 'resnet50']:
            self.backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.fc.in_features
        # mobile net-based
        elif backbone_name in ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']:
            self.backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.classifier[-1].in_features
        # ViT based
        elif backbone_name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
            self.backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.heads[-1].in_features
        else:
            raise ValueError(f"Backbone {backbone_name} not supported yet.")

        for param in self.backbone.parameters():
            param.requires_grad = False

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


        self.backbone.fc = nn.Identity()

        # Projection / Embedding Head
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True)
        )

    def forward_once(self, x):
        features = self.backbone(x)
        embeddings = self.fc(features)
        return embeddings

    def forward(self, input1, input2):
        v1 = self.forward_once(input1)
        v2 = self.forward_once(input2)
        return v1, v2