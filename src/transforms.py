import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class SiameseAffine:
    """
    Implements the paper's affine distortions:
    1. Stochastically choose parameters (prob=0.5 for each).
    2. Combine them into ONE transformation matrix.
    3. Apply ONCE to prevent multiple interpolation blurs.
    """

    def __init__(self):
        self.prob = 0.5
        # Rotation: [-10, 10] degrees
        self.rot_limit = 10.0
        # Shear: [-0.3, 0.3] radians is approx [-17.18, 17.18] degrees
        self.shear_limit = 17.18
        # Scale: [0.8, 1.2]
        self.scale_range = (0.8, 1.2)
        # Translation: [-2, 2] pixels (on 105x105).
        # For 250x250, we use relative %: 2/105 approx 0.02
        self.trans_limit = 0.02

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