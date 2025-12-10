import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.backbone_name = backbone_name

        if backbone_name in ['resnet18', 'resnet34', 'resnet50']:
            self.backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.fc.in_features
            
        elif backbone_name in ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']:
            self.backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None)
            # MobileNet classifier is usually a Sequential
            in_features = self.backbone.classifier[-1].in_features
            
        elif backbone_name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
            self.backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None)
            # ViT heads is a Sequential
            in_features = self.backbone.heads[-1].in_features
        else:
            raise ValueError(f"Backbone {backbone_name} not supported yet.")

        # Freeze weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Modify First Layer
        if 'resnet' in backbone_name:
            original_layer = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels=1, 
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=False
            )
            with torch.no_grad():
                self.backbone.conv1.weight.data = original_layer.weight.data.mean(dim=1, keepdim=True)
            
            # Remove Head
            self.backbone.fc = nn.Identity()

        elif 'mobilenet' in backbone_name:
            # MobileNet v2/v3 first layer is 'features[0][0]'
            original_layer = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=1,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=False
            )
            with torch.no_grad():
                self.backbone.features[0][0].weight.data = original_layer.weight.data.mean(dim=1, keepdim=True)
            
            # Remove Head
            self.backbone.classifier = nn.Identity()

        elif 'vit' in backbone_name:
            # ViT first layer is 'conv_proj' (Patch Embedding)
            original_layer = self.backbone.conv_proj
            self.backbone.conv_proj = nn.Conv2d(
                in_channels=1,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=original_layer.bias
            )
            with torch.no_grad():
                self.backbone.conv_proj.weight.data = original_layer.weight.data.mean(dim=1, keepdim=True)

            # Remove Head
            self.backbone.heads = nn.Identity()

        # Projection Head
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