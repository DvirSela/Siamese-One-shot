import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns

from src.config import BATCH_SIZE, SEED
from src.consts import IMG_DIR, DEVICE, TEST_PAIRS_FILE, MODEL_PATH
from src.dataset import SiameseDataset, parse_lfw_pairs
from src.models.siamese_model import SiameseNetwork
from src.utils import set_seed
from src.transforms import get_transforms


def validate(model, val_loader, criterion) -> Tuple[float, float]:
    """
    Validate the model on the validation dataset.
    Args:
        model (nn.Module): The Siamese Network model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion: Loss function.
    Returns:
        Tuple[float, float]: Average loss and accuracy on validation set.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(
                DEVICE), labels.to(DEVICE).unsqueeze(1)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * img1.size(0)

            predicted = (outputs > 0.5).float()
            total_correct += (predicted == labels).sum().item()
            total_samples += img1.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def inverse_transform(img_tensor) -> np.ndarray:
    """
    Convert tensor back to numpy image for plotting.
    Args:
        img_tensor (Tensor): Image tensor.
    Returns:
        np.ndarray: Numpy array image.
    """
    img = img_tensor.squeeze().cpu().numpy()
    return img


def visualize_hardest_errors(hardest_pos, hardest_neg, save_path="error_analysis.png"):
    """
    Visualizes the pair that the model was MOST wrong about.
    Args:
        hardest_pos (tuple): (img1, img2, score, label) for hardest positive pair (False Negative).
        hardest_neg (tuple): (img1, img2, score, label) for hardest negative pair (False Positive).
        save_path (str): Path to save the visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    if hardest_pos:
        img1, img2, score, label = hardest_pos
        axes[0, 0].imshow(inverse_transform(img1), cmap='gray')
        axes[0, 0].set_title("Hardest False Negative (Img A)")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(inverse_transform(img2), cmap='gray')
        axes[0, 1].set_title(f"Actual: Same | Pred Score: {score:.4f}")
        axes[0, 1].axis('off')

    if hardest_neg:
        img1, img2, score, label = hardest_neg
        axes[1, 0].imshow(inverse_transform(img1), cmap='gray')
        axes[1, 0].set_title("Hardest False Positive (Img A)")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(inverse_transform(img2), cmap='gray')
        axes[1, 1].set_title(f"Actual: Diff | Pred Score: {score:.4f}")
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved error analysis to {save_path}")
    plt.show()


def plot_score_distribution(labels, scores, save_path="score_dist.png"):
    """
    Plots a histogram of scores for positive and negative pairs.
    """
    plt.figure(figsize=(10, 6))

    # Convert to numpy arrays if they aren't already
    labels = np.array(labels)
    scores = np.array(scores)

    # Separate scores
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    # Plot histograms
    plt.hist(pos_scores, bins=50, alpha=0.5,
             label='Positive Pairs (Same)', color='green')
    plt.hist(neg_scores, bins=50, alpha=0.5,
             label='Negative Pairs (Diff)', color='red')

    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score (0=Diff, 1=Same)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    print(f"Saved score distribution plot to {save_path}")
    plt.show()


def evaluate():
    """
    Evaluates the Siamese Network on the test dataset and visualizes the hardest errors.
    """
    set_seed(SEED)
    print(f"Evaluating on device: {DEVICE}")
    print(f"Loading Test Data from {TEST_PAIRS_FILE}...")

    test_pairs, test_labels = parse_lfw_pairs(TEST_PAIRS_FILE, IMG_DIR)

    test_dataset = SiameseDataset(
        test_pairs, test_labels, transform=get_transforms(is_train=False))
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading Model...")
    model = SiameseNetwork().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Weights loaded successfully.")
    else:
        print("Error: best_model.pth not found!")
        return

    model.eval()

    total = 0
    correct = 0

    all_scores = []
    all_labels = []

    min_pos_score = 1.0
    hardest_fn_pair = None
    max_neg_score = 0.0
    hardest_fp_pair = None

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(
                DEVICE), labels.to(DEVICE).unsqueeze(1)

            outputs = model(img1, img2)
            predicted = (outputs > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            scores_np = outputs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            all_scores.extend(scores_np)
            all_labels.extend(labels_np)

            for i in range(len(labels_np)):
                score = scores_np[i]
                label = labels_np[i]

                if label == 1.0:
                    if score < min_pos_score:
                        min_pos_score = score
                        hardest_fn_pair = (
                            img1[i].cpu(), img2[i].cpu(), score, label)

                if label == 0.0:
                    if score > max_neg_score:
                        max_neg_score = score
                        hardest_fp_pair = (
                            img1[i].cpu(), img2[i].cpu(), score, label)

    accuracy = correct / total
    print("-" * 30)
    print(f"Test Accuracy: {accuracy:.2%}")

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("-" * 30)
    plot_score_distribution(all_labels, all_scores)

    visualize_hardest_errors(hardest_fn_pair, hardest_fp_pair)


if __name__ == "__main__":
    evaluate()
