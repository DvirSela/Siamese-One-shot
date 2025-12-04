import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class SiameseAffine:
    def __init__(self):
        self.prob = 0.5 # Kept prob 0.5 to not destroy too many images
        # INCREASED ROTATION: 10 -> 20 degrees
        self.rot_limit = 20.0 
        # INCREASED SHEAR: ~17 -> ~25 degrees
        self.shear_limit = 25.0
        # INCREASED SCALE: 0.8-1.2 -> 0.7-1.3
        self.scale_range = (0.7, 1.3)
        # INCREASED TRANSLATION: 2% -> 5%
        self.trans_limit = 0.05

    def __call__(self, img):

        # Rotation
        angle = random.uniform(-self.rot_limit,
                               self.rot_limit) if random.random() < self.prob else 0.0

        # Shear (X and Y independently as per paper)
        shear_x = random.uniform(-self.shear_limit,
                                 self.shear_limit) if random.random() < self.prob else 0.0
        shear_y = random.uniform(-self.shear_limit,
                                 self.shear_limit) if random.random() < self.prob else 0.0
        shear = [shear_x, shear_y]

        # Scale
        scale = random.uniform(
            *self.scale_range) if random.random() < self.prob else 1.0

        # Translation
        tx = 0
        ty = 0
        if random.random() < self.prob:
            w, h = img.size
            max_dx = self.trans_limit * w
            max_dy = self.trans_limit * h
            tx = random.uniform(-max_dx, max_dx)
            ty = random.uniform(-max_dy, max_dy)
        translate = [tx, ty]

        return F.affine(
            img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=transforms.InterpolationMode.BILINEAR
        )

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            SiameseAffine(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])