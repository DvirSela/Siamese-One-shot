import os
from typing import List, Tuple
import random

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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


def split_pairs_by_identity(pairs, labels, val_size=0.2, seed=42) -> Tuple[Tuple[List[Tuple[str, str]], List[int]], Tuple[List[Tuple[str, str]], List[int]]]:
    """
    Splits pairs into Train and Validation sets ensuring NO overlapping identities.

    Args:
        pairs (list): List of tuples [(path1, path2), ...]
        labels (list): List of labels [1, 0, ...]
        val_size (float): Percentage of identities to hold out.
        seed (int): the seed
    Returns:
        (train_pairs, train_labels), (val_pairs, val_labels)
        train_pairs: List of tuples for training set
        train_labels: Corresponding labels for training set
        val_pairs: List of tuples for validation set
        val_labels: Corresponding labels for validation set
    """
    # 1. Extract all unique identities from the pairs
    # We assume path format is: .../Name/Name_0001.jpg
    # So we split by separator and take the parent folder name
    all_names = set()
    for p1, p2 in pairs:
        # Extract name from path (depends on your OS separator, usually / or \)
        name1 = p1.split(os.sep)[-2]
        name2 = p2.split(os.sep)[-2]
        all_names.add(name1)
        all_names.add(name2)

    all_names = list(all_names)
    print(f"Total unique identities in dataset: {len(all_names)}")

    # 2. Split the NAMES (not the pairs)
    train_names, val_names = train_test_split(
        all_names, test_size=val_size, random_state=seed
    )

    # Convert to sets for O(1) lookup
    train_names_set = set(train_names)
    val_names_set = set(val_names)

    # 3. Bucket the PAIRS based on the names
    train_pairs_clean = []
    train_labels_clean = []

    val_pairs_clean = []
    val_labels_clean = []

    dropped_count = 0

    for i, (p1, p2) in enumerate(pairs):
        name1 = p1.split(os.sep)[-2]
        name2 = p2.split(os.sep)[-2]

        # Logic:
        # If BOTH names are in Train Set -> Add to Train
        # If BOTH names are in Val Set   -> Add to Val
        # If one is Train and one is Val -> DROP IT (Leakage risk)

        if name1 in train_names_set and name2 in train_names_set:
            train_pairs_clean.append((p1, p2))
            train_labels_clean.append(labels[i])

        elif name1 in val_names_set and name2 in val_names_set:
            val_pairs_clean.append((p1, p2))
            val_labels_clean.append(labels[i])

        else:
            dropped_count += 1

    print(f"Split Summary:")
    print(f"  Train Pairs: {len(train_pairs_clean)}")
    print(f"  Val Pairs:   {len(val_pairs_clean)}")
    print(f"  Dropped:     {dropped_count} (Mixed pairs)")

    return (train_pairs_clean, train_labels_clean), (val_pairs_clean, val_labels_clean)


def generate_new_negatives(names_list, quantity, img_dir) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Generates new negative pairs from a list of allowed names.
    Args:
        names_list (list): List of names (identities) to sample from.
        quantity (int): Number of negative pairs to generate.
        img_dir (str): Directory where images are stored.
    Returns:
        new_pairs (list): List of newly generated negative pairs.
        new_labels (list): List of labels (all 0s for negatives).
    """
    new_pairs = []
    new_labels = []

    name_to_imgs = {}
    valid_names = []

    for name in names_list:
        path = os.path.join(img_dir, name)
        if os.path.isdir(path):
            imgs = [os.path.join(path, f)
                    for f in os.listdir(path) if f.endswith('.jpg')]
            if len(imgs) > 0:
                name_to_imgs[name] = imgs
                valid_names.append(name)

    count = 0
    while count < quantity:
        # Pick two random DIFFERENT people
        n1, n2 = random.sample(valid_names, 2)

        # Pick a random image for each
        img1 = random.choice(name_to_imgs[n1])
        img2 = random.choice(name_to_imgs[n2])

        new_pairs.append((img1, img2))
        new_labels.append(0)  # Label 0 for negative
        count += 1

    return new_pairs, new_labels

class SiameseDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int], transform=None):
        """
        Modified to accept raw lists instead of file paths.
        """
        self.pairs = pairs
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __getitem__(self, index):
        img1_path, img2_path = self.pairs[index]
        label = self.labels[index]

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.pairs)

def get_train_val_datasets(pairs_file, img_dir, val_size=0.2, transform_train=None, transform_val=None) -> Tuple[SiameseDataset, SiameseDataset]:
    """
    Prepares the Siamese Train and Validation datasets with proper splitting and balancing.
    Args:
        pairs_file (str): Path to the pairs.txt file.
        img_dir (str): Directory where images are stored.
        val_size (float): Proportion of identities to use for validation.
        transform_train: Transformations for training data.
        transform_val: Transformations for validation data.
    Returns:
        train_dataset (SiameseDataset): Training dataset.
    """
    print("Parsing and Splitting Data...")
    all_train_pairs, all_train_labels = parse_lfw_pairs(pairs_file, img_dir)

    original_count = len(all_train_pairs)
    
    (train_pairs, train_labels), (val_pairs, val_labels) = split_pairs_by_identity(
        all_train_pairs, all_train_labels, val_size=val_size
    )
    
    current_count = len(train_pairs) + len(val_pairs)
    lost_pairs = original_count - current_count

    # I backfill the lost pairs as negatives in the TRAIN set
    if lost_pairs > 0:
        print(f"Recovering {lost_pairs} dropped pairs by generating new Training negatives...")
        
        train_names_set = set()
        for p1, p2 in train_pairs:
            train_names_set.add(p1.split(os.sep)[-2])
            train_names_set.add(p2.split(os.sep)[-2])
            
        new_train_p, new_train_l = generate_new_negatives(
            list(train_names_set), lost_pairs, img_dir
        )
        
        train_pairs.extend(new_train_p)
        train_labels.extend(new_train_l)

    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos

    if val_pos > val_neg:
        needed = int(val_pos - val_neg)
        print(f"Balancing Val Set: Generating {needed} new negative pairs...")

        val_names_set = set()
        for p1, p2 in val_pairs:
            val_names_set.add(p1.split(os.sep)[-2])
            val_names_set.add(p2.split(os.sep)[-2])

        new_p, new_l = generate_new_negatives(
            list(val_names_set), needed, img_dir)
        val_pairs.extend(new_p)
        val_labels.extend(new_l)

    print(f"Final Dataset: Train {len(train_pairs)} pairs | Val {len(val_pairs)} pairs")
    print(f"  Train Positives: {sum(train_labels)} | Train Negatives: {len(train_labels) - sum(train_labels)}")
    print(f"  Val Positives: {sum(val_labels)} | Val Negatives: {len(val_labels) - sum(val_labels)}")

    train_dataset = SiameseDataset(
        train_pairs, train_labels, transform=transform_train)
    val_dataset = SiameseDataset(
        val_pairs, val_labels, transform=transform_val)

    return train_dataset, val_dataset