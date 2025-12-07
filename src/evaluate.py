import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from src.config import BATCH_SIZE, SEED, THRESHOLD, MODEL_NAME
from src.consts import IMG_DIR, DEVICE, TEST_PAIRS_FILE, MODEL_PATH
from src.dataset import SiameseDataset, parse_lfw_pairs
from src.models.siamese_model import SiameseNetwork
from src.utils import set_seed
from src.transforms import get_transforms


def validate(model, val_loader, criterion):
    """
    Simple validation loop for training progress monitoring.
    Uses a fixed threshold (from config) to calculate accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    threshold = THRESHOLD

    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(
                DEVICE), labels.to(DEVICE)

            # Forward
            v1, v2 = model(img1, img2)

            # Loss
            loss = criterion(v1, v2, labels.squeeze())
            total_loss += loss.item() * img1.size(0)

            dist = torch.nn.functional.pairwise_distance(v1, v2)
            predicted = (dist < threshold).float()

            total_correct += (predicted == labels.squeeze()).sum().item()
            total_samples += img1.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def inverse_transform(img_tensor) -> np.ndarray:
    """
    Convert tensor back to numpy image for plotting.
    """
    img = img_tensor.squeeze().cpu().numpy()
    return img


def visualize_hardest_errors(hardest_pos, hardest_neg, save_path="error_analysis.png"):
    """
    Visualizes the pair that the model was MOST wrong about based on distance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # 1. Hardest False Negative (Actual: Same, Pred: Different)
    # This means the Distance was TOO HIGH (Model thought they were different).
    if hardest_pos:
        img1, img2, dist, label = hardest_pos
        axes[0, 0].imshow(inverse_transform(img1), cmap='gray')
        axes[0, 0].set_title("Hardest False Negative (Img A)")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(inverse_transform(img2), cmap='gray')
        axes[0, 1].set_title(f"Actual: Same | Dist: {dist:.4f}")
        axes[0, 1].axis('off')

    # 2. Hardest False Positive (Actual: Different, Pred: Same)
    # This means the Distance was TOO LOW (Model thought they were same).
    if hardest_neg:
        img1, img2, dist, label = hardest_neg
        axes[1, 0].imshow(inverse_transform(img1), cmap='gray')
        axes[1, 0].set_title("Hardest False Positive (Img A)")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(inverse_transform(img2), cmap='gray')
        axes[1, 1].set_title(f"Actual: Diff | Dist: {dist:.4f}")
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved error analysis to {save_path}")
    plt.show()


def plot_distance_distribution(labels, distances, threshold, save_path="dist_dist.png"):
    """
    Plots a histogram of distances for positive and negative pairs.
    """
    plt.figure(figsize=(10, 6))

    labels = np.array(labels)
    distances = np.array(distances)

    pos_dists = distances[labels == 1]
    neg_dists = distances[labels == 0]

    plt.hist(pos_dists, bins=50, alpha=0.5,
             label='Positive Pairs (Same)', color='green')
    plt.hist(neg_dists, bins=50, alpha=0.5,
             label='Negative Pairs (Diff)', color='red')

    plt.axvline(x=threshold, color='black', linestyle='--',
                label=f'Best Threshold ({threshold:.2f})')
    plt.title('Distribution of Embedding Distances')
    plt.xlabel('Euclidean Distance (Lower = More Similar)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    print(f"Saved distance distribution plot to {save_path}")
    plt.show()


def find_optimal_threshold(labels, distances):
    """
    Finds the threshold that maximizes accuracy by scanning the ACTUAL range of distances.
    """
    best_acc = 0.0
    best_thresh = 0.0

    # 1. Determine the search range dynamically
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    print(f"Scanning thresholds between {min_dist:.4f} and {max_dist:.4f}...")

    # 2. Create 1000 evenly spaced thresholds across the ACTUAL range
    thresholds = np.linspace(min_dist, max_dist, num=1000)

    for thresh in thresholds:
        # Predict 1 (Same) if distance < threshold
        predictions = (distances < thresh).astype(int)
        accuracy = (predictions == labels).mean()

        if accuracy > best_acc:
            best_acc = accuracy
            best_thresh = thresh

    return best_acc, best_thresh


def evaluate():
    set_seed(SEED)
    print(f"Evaluating on device: {DEVICE}")
    print(f"Loading Test Data from {TEST_PAIRS_FILE}...")

    test_pairs, test_labels = parse_lfw_pairs(TEST_PAIRS_FILE, IMG_DIR)

    test_dataset = SiameseDataset(
        test_pairs, test_labels, transform=get_transforms(is_train=False))
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading Model...")
    # model = SiameseNetwork().to(DEVICE)
    model = SiameseNetwork(backbone_name=MODEL_NAME,
                           pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Weights loaded successfully.")
    else:
        print("Error: best_model.pth not found!")
        return

    model.eval()

    all_dists = []
    all_labels = []

    print("Calculating distances...")
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(
                DEVICE), labels.to(DEVICE).unsqueeze(1)

            # Get Vectors
            v1, v2 = model(img1, img2)

            # Calculate Distance (Euclidean)
            dists = F.pairwise_distance(v1, v2)

            all_dists.extend(dists.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    # 2. Find Best Threshold
    best_acc, best_thresh = find_optimal_threshold(all_labels, all_dists)

    print("-" * 30)
    print(
        f"Best Test Accuracy: {best_acc:.2%} at Threshold: {best_thresh:.2f}")

    fpr, tpr, thresholds = roc_curve(all_labels, -all_dists)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("-" * 30)

    # 4. Plotting
    plot_distance_distribution(all_labels, all_dists, best_thresh)

    # 5. Find Hardest Errors using the BEST threshold
    max_pos_dist = -1.0
    hardest_fn_pair = None  # Should have been predicted 1, but dist was huge

    min_neg_dist = float('inf')
    hardest_fp_pair = None  # Should have been predicted 0, but dist was tiny

    # We need to re-iterate or store images to visualize.
    # For simplicity, we loop again quickly or if dataset is small.
    # Since we need the IMAGES to plot, we do a quick pass or intelligent selection.
    # To save memory, we'll just grab them on a second quick pass knowing the threshold.

    print("Mining hardest examples...")
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(
                DEVICE), labels.to(DEVICE).unsqueeze(1)
            v1, v2 = model(img1, img2)
            dists = F.pairwise_distance(v1, v2)

            dists_np = dists.cpu().numpy()
            labels_np = labels.cpu().numpy().flatten()

            for i in range(len(labels_np)):
                dist = dists_np[i]
                label = labels_np[i]

                # False Negative: Same Person (1), but Distance > Threshold
                if label == 1.0:
                    if dist > max_pos_dist:
                        max_pos_dist = dist
                        hardest_fn_pair = (
                            img1[i].cpu(), img2[i].cpu(), dist, label)

                # False Positive: Diff Person (0), but Distance < Threshold
                if label == 0.0:
                    if dist < min_neg_dist:
                        min_neg_dist = dist
                        hardest_fp_pair = (
                            img1[i].cpu(), img2[i].cpu(), dist, label)

    visualize_hardest_errors(hardest_fn_pair, hardest_fp_pair)


if __name__ == "__main__":
    evaluate()
