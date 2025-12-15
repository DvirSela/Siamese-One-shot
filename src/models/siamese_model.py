import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import clip

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.backbone_name = backbone_name

        if backbone_name == 'clip_vit_b32':
            model, _ = clip.load("ViT-B/32", device='cpu', jit=False)
            self.backbone = model.visual.float()
            in_features = 512 
            
        elif backbone_name in ['resnet18', 'resnet34', 'resnet50']:
            self.backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.fc.in_features
            
        elif backbone_name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
            self.backbone = getattr(models, backbone_name)(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.heads[-1].in_features
        else:
            raise ValueError(f"Backbone {backbone_name} not supported yet.")

        for param in self.backbone.parameters():
            param.requires_grad = False

        if backbone_name == 'clip_vit_b32':
            original_layer = self.backbone.conv1
            
            self.backbone.conv1 = nn.Conv2d(
                in_channels=1, 
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=0, # CLIP uses 0 padding usually
                bias=False
            )
            
            with torch.no_grad():
                self.backbone.conv1.weight.data = original_layer.weight.data.mean(dim=1, keepdim=True)
                
        elif 'resnet' in backbone_name:
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
            self.backbone.fc = nn.Identity()

        elif 'vit' in backbone_name and 'clip' not in backbone_name:
            original_layer = self.backbone.conv_proj
            has_bias = original_layer.bias is not None
            self.backbone.conv_proj = nn.Conv2d(
                in_channels=1,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=has_bias
            )
            with torch.no_grad():
                self.backbone.conv_proj.weight.data = original_layer.weight.data.mean(dim=1, keepdim=True)
                if has_bias:
                    self.backbone.conv_proj.bias.data = original_layer.bias.data
            self.backbone.heads = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
        )

    def forward_once(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['last_hidden_state'][:, 0]
            
        embeddings = self.fc(features)
        embeddings = F.normalize(embeddings, p=2, dim=1) 
        return embeddings

    def forward(self, input1, input2):
        v1 = self.forward_once(input1)
        v2 = self.forward_once(input2)
        return v1, v2