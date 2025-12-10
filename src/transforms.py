import random
import torch
import torchvision.transforms as transforms

class SiameseAffine:
    def __init__(self, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img):
        return transforms.RandomAffine(
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear
        )(img)

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            
            # 1. Resize strictly for ViT
            transforms.Resize((224, 224)), 
            
            # 2. Geometric Augmentations
            SiameseAffine(),
            
            # 3. Blur (Robustness)
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            
            transforms.ToTensor(),
            
            # Normalization
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            
            # Resize Validation/Test data too!
            transforms.Resize((224, 224)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])