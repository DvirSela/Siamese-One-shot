import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.consts import SAME_PERSON_LABEL, DIFFERENT_PERSON_LABEL

def parse_lfw_pairs(pairs_filepath, img_dir_path) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Parses the LFW pairs file and generates file paths and labels.
    Args:
        pairs_filepath (str): Path to the pairs.txt file.
        img_dir_path (str): Directory where images are stored.
    Raises:
        FileNotFoundError: If the pairs file does not exist.
    Returns:
        pairs (List[Tuple[str, str]]): List of tuples containing image file paths.
        labels (List[int]): List of labels (1 for same, 0 for different).
    """
    pairs = []
    labels = []

    if not os.path.exists(pairs_filepath):
        raise FileNotFoundError(f"Pairs file not found: {pairs_filepath}")

    with open(pairs_filepath, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        components = line.strip().split()

        if len(components) == 3:
            # Same Person
            name = components[0]
            id1 = components[1]
            id2 = components[2]
            img1_path = os.path.join(
                img_dir_path, name, f"{name}_{id1.zfill(4)}.jpg")
            img2_path = os.path.join(
                img_dir_path, name, f"{name}_{id2.zfill(4)}.jpg")
            pairs.append((img1_path, img2_path))
            labels.append(SAME_PERSON_LABEL)

        elif len(components) == 4:
            # Different People
            name1 = components[0]
            id1 = components[1]
            name2 = components[2]
            id2 = components[3]
            img1_path = os.path.join(
                img_dir_path, name1, f"{name1}_{id1.zfill(4)}.jpg")
            img2_path = os.path.join(
                img_dir_path, name2, f"{name2}_{id2.zfill(4)}.jpg")
            pairs.append((img1_path, img2_path))
            labels.append(DIFFERENT_PERSON_LABEL)

    return pairs, labels


class SiameseDataset(Dataset):
    def __init__(self, pairs_file, img_dir, transform=None):
        """
        Args:
            pairs_file (string): Path to the pairs file.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.pairs, self.labels = parse_lfw_pairs(pairs_file, img_dir)
        self.transform = transform

        # Convert labels to float for BCE Loss
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __getitem__(self, index):
        img1_path, img2_path = self.pairs[index]
        label = self.labels[index]

        # Load Images (Grayscale 'L')
        # We assume images exist. If not, this will throw an error,
        # which is good for debugging paths early.
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.pairs)
