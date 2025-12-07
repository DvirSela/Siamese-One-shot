import os
import random
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

from src.consts import SAME_PERSON_LABEL, DIFFERENT_PERSON_LABEL



class LabeledDataset(Dataset):
    """
    Returns individual images with their person ID (label).
    Used for Batch Hard Training.
    """
    def __init__(self, pairs, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. Build Dictionary: Name -> List of Images
        self.people_dict = defaultdict(list)
        for p1, p2 in pairs:
            name1 = os.path.basename(os.path.dirname(p1))
            name2 = os.path.basename(os.path.dirname(p2))
            self.people_dict[name1].append(p1)
            self.people_dict[name2].append(p2)
            
        # Deduplicate
        for name in self.people_dict:
            self.people_dict[name] = list(set(self.people_dict[name]))
            
        # 2. Assign Integer IDs to names
        # Filter for people with at least K=2 images
        self.valid_names = [n for n in self.people_dict.keys() if len(self.people_dict[n]) >= 2]
        self.name_to_id = {name: i for i, name in enumerate(self.valid_names)}
        
        # 3. Flatten to List of (Image, LabelID)
        self.data = []
        for name in self.valid_names:
            label = self.name_to_id[name]
            for img_path in self.people_dict[name]:
                self.data.append((img_path, label))
                
    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

class PKSampler(Sampler):
    """
    Randomly samples batches ensuring P identities with K images each.
    """
    def __init__(self, dataset, p=8, k=4):
        self.dataset = dataset
        self.p = p # Number of people per batch
        self.k = k # Number of images per person
        self.batch_size = p * k
        
        # Organize indices by label
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.data):
            self.label_to_indices[label].append(idx)
            
        # Valid labels (must have at least K images)
        self.labels = list(self.label_to_indices.keys())
        
        # Calculate length roughly
        self.num_batches = len(dataset) // self.batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            # 1. Pick P random people
            selected_labels = random.sample(self.labels, self.p)
            
            batch_indices = []
            for label in selected_labels:
                # 2. Pick K random images for each person
                # (Use replacement if a person has fewer than K images, though we filtered for >=2)
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    selected = random.sample(indices, self.k)
                else:
                    selected = np.random.choice(indices, self.k, replace=True).tolist()
                batch_indices.extend(selected)
                
            yield batch_indices

    def __len__(self):
        return self.num_batches

# UPDATE FACTORY
def get_ohem_dataloaders(pairs_file, img_dir, val_size=0.2, transform_train=None, transform_val=None, p=8, k=4):
    print("Parsing and Splitting Data for OHEM...")
    all_train_pairs, all_train_labels = parse_lfw_pairs(pairs_file, img_dir)

    # 1. Leak-Free Split
    (train_pairs, train_labels), (val_pairs, val_labels) = split_pairs_by_identity(
        all_train_pairs, all_train_labels, val_size=val_size
    )
    
    # 2. Create OHEM Train Dataset (Labeled)
    print("Creating Labeled Train Dataset...")
    train_dataset = LabeledDataset(train_pairs, img_dir, transform=transform_train)
    
    # 3. Create Sampler
    train_sampler = PKSampler(train_dataset, p=p, k=k)
    
    # 4. Validation remains Pair-based (Standard Metric)
    # We still need to balance it!
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    if val_pos > val_neg:
        needed = int(val_pos - val_neg)
        val_names_set = set()
        for p1, p2 in val_pairs:
            val_names_set.add(os.path.basename(os.path.dirname(p1)))
            val_names_set.add(os.path.basename(os.path.dirname(p2)))
        new_p, new_l = generate_new_negatives(list(val_names_set), needed, img_dir)
        val_pairs.extend(new_p)
        val_labels.extend(new_l)
        
    val_dataset = SiameseDataset(val_pairs, val_labels, transform=transform_val)

    return train_dataset, train_sampler, val_dataset




def parse_lfw_pairs(pairs_filepath, img_dir_path) -> Tuple[List[Tuple[str, str]], List[int]]:
    """Parses the LFW pairs file and generates file paths and labels."""
    pairs = []
    labels = []

    if not os.path.exists(pairs_filepath):
        raise FileNotFoundError(f"Pairs file not found: {pairs_filepath}")

    with open(pairs_filepath, 'r') as f:
        lines = f.readlines()

    # Skip header line containing the number of pairs
    for line in lines[1:]:
        components = line.strip().split()

        if len(components) == 3:
            # Same Person
            name = components[0]
            id1 = components[1]
            id2 = components[2]
            img1_path = os.path.join(img_dir_path, name, f"{name}_{id1.zfill(4)}.jpg")
            img2_path = os.path.join(img_dir_path, name, f"{name}_{id2.zfill(4)}.jpg")
            pairs.append((img1_path, img2_path))
            labels.append(SAME_PERSON_LABEL)

        elif len(components) == 4:
            # Different People
            name1 = components[0]
            id1 = components[1]
            name2 = components[2]
            id2 = components[3]
            img1_path = os.path.join(img_dir_path, name1, f"{name1}_{id1.zfill(4)}.jpg")
            img2_path = os.path.join(img_dir_path, name2, f"{name2}_{id2.zfill(4)}.jpg")
            pairs.append((img1_path, img2_path))
            labels.append(DIFFERENT_PERSON_LABEL)

    return pairs, labels


def split_pairs_by_identity(pairs, labels, val_size=0.2, seed=42):
    """Splits pairs into Train and Validation sets ensuring NO overlapping identities."""
    all_names = set()
    for p1, p2 in pairs:
        name1 = os.path.basename(os.path.dirname(p1))
        name2 = os.path.basename(os.path.dirname(p2))
        all_names.add(name1)
        all_names.add(name2)

    all_names = list(all_names)
    print(f"Total unique identities in dataset: {len(all_names)}")

    train_names, val_names = train_test_split(all_names, test_size=val_size, random_state=seed)
    
    train_names_set = set(train_names)
    val_names_set = set(val_names)

    train_pairs_clean = []
    train_labels_clean = []
    val_pairs_clean = []
    val_labels_clean = []
    dropped_count = 0

    for i, (p1, p2) in enumerate(pairs):
        name1 = os.path.basename(os.path.dirname(p1))
        name2 = os.path.basename(os.path.dirname(p2))

        if name1 in train_names_set and name2 in train_names_set:
            train_pairs_clean.append((p1, p2))
            train_labels_clean.append(labels[i])
        elif name1 in val_names_set and name2 in val_names_set:
            val_pairs_clean.append((p1, p2))
            val_labels_clean.append(labels[i])
        else:
            dropped_count += 1

    print(f"Split Summary: Train Pairs: {len(train_pairs_clean)} | Val Pairs: {len(val_pairs_clean)} | Dropped: {dropped_count}")
    return (train_pairs_clean, train_labels_clean), (val_pairs_clean, val_labels_clean)


def generate_new_negatives(names_list, quantity, img_dir):
    """Generates new negative pairs from a list of allowed names."""
    new_pairs = []
    new_labels = []
    
    name_to_imgs = {}
    valid_names = []
    
    for name in names_list:
        path = os.path.join(img_dir, name)
        if os.path.isdir(path):
            imgs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
            if len(imgs) > 0:
                name_to_imgs[name] = imgs
                valid_names.append(name)
    
    if len(valid_names) < 2:
        return [], []

    count = 0
    while count < quantity:
        n1, n2 = random.sample(valid_names, 2)
        img1 = random.choice(name_to_imgs[n1])
        img2 = random.choice(name_to_imgs[n2])
        new_pairs.append((img1, img2))
        new_labels.append(0) 
        count += 1
        
    return new_pairs, new_labels


class SiameseDataset(Dataset):
    """Standard Pair-based dataset for Validation/Testing."""
    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int], transform=None):
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


class TripletSiameseDataset(Dataset):
    """Triplet-based dataset for Training (Anchor, Positive, Negative)."""
    def __init__(self, pairs, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Build dictionary from the provided pairs list (which is already split safe)
        self.people_dict = {}
        self.people_list = []
        
        for p1, p2 in pairs:
            name1 = os.path.basename(os.path.dirname(p1))
            name2 = os.path.basename(os.path.dirname(p2))
            
            if name1 not in self.people_dict:
                self.people_dict[name1] = []
                self.people_list.append(name1)
            self.people_dict[name1].append(p1)
            
            if name2 not in self.people_dict:
                self.people_dict[name2] = []
                self.people_list.append(name2)
            self.people_dict[name2].append(p2)
            
        # Deduplicate
        for name in self.people_dict:
            self.people_dict[name] = list(set(self.people_dict[name]))
            
        # Filter (must have >= 2 images to form a positive pair)
        self.people_list = [p for p in self.people_list if len(self.people_dict[p]) > 1]
        print(f"Triplet Dataset: Found {len(self.people_list)} identities with 2+ images.")

    def __getitem__(self, index):
        anchor_name = self.people_list[index]
        anchor_imgs = self.people_dict[anchor_name]
        
        # Positive
        if len(anchor_imgs) == 2:
            anchor_path, pos_path = anchor_imgs[0], anchor_imgs[1]
        else:
            anchor_path, pos_path = random.sample(anchor_imgs, 2)
            
        # Negative
        neg_name = anchor_name
        while neg_name == anchor_name:
            neg_name = random.choice(self.people_list)
        neg_path = random.choice(self.people_dict[neg_name])
        
        # Load
        anchor_img = Image.open(anchor_path).convert("L")
        pos_img = Image.open(pos_path).convert("L")
        neg_img = Image.open(neg_path).convert("L")
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
            
        return anchor_img, pos_img, neg_img

    def __len__(self):
        return len(self.people_list)


def get_dataloaders(pairs_file, img_dir, val_size=0.2, transform_train=None, transform_val=None, use_triplet=False):
    """
    Returns (train_dataset, val_dataset) guaranteeing no leakage.
    """
    print("Parsing and Splitting Data...")
    all_train_pairs, all_train_labels = parse_lfw_pairs(pairs_file, img_dir)
    original_count = len(all_train_pairs)

    # 1. Leak-Free Split
    (train_pairs, train_labels), (val_pairs, val_labels) = split_pairs_by_identity(
        all_train_pairs, all_train_labels, val_size=val_size
    )
    
    # 2. Backfill Training Pairs (Important for volume, even if using Triplets we use pairs to derive names)
    current_count = len(train_pairs) + len(val_pairs)
    lost_pairs = original_count - current_count
    
    if lost_pairs > 0:
        print(f"Recovering {lost_pairs} dropped pairs by generating new Training negatives...")
        train_names_set = set()
        for p1, p2 in train_pairs:
            train_names_set.add(os.path.basename(os.path.dirname(p1)))
            train_names_set.add(os.path.basename(os.path.dirname(p2)))
            
        new_train_p, new_train_l = generate_new_negatives(list(train_names_set), lost_pairs, img_dir)
        train_pairs.extend(new_train_p)
        train_labels.extend(new_train_l)
    
    # 3. Balance Validation (CRITICAL)
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    if val_pos > val_neg:
        needed = int(val_pos - val_neg)
        print(f"Balancing Val Set: Generating {needed} new negative pairs...")
        val_names_set = set()
        for p1, p2 in val_pairs:
            val_names_set.add(os.path.basename(os.path.dirname(p1)))
            val_names_set.add(os.path.basename(os.path.dirname(p2)))
        new_p, new_l = generate_new_negatives(list(val_names_set), needed, img_dir)
        val_pairs.extend(new_p)
        val_labels.extend(new_l)
    
    print(f"Final Split: Train {len(train_pairs)} pairs | Val {len(val_pairs)} pairs")

    if use_triplet:
        print("Creating Triplet Train Dataset...")
        train_dataset = TripletSiameseDataset(train_pairs, img_dir, transform=transform_train)
    else:
        train_dataset = SiameseDataset(train_pairs, train_labels, transform=transform_train)
        
    val_dataset = SiameseDataset(val_pairs, val_labels, transform=transform_val)

    return train_dataset, val_dataset