import os
from typing import Tuple, List
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

def validate(model: torch.nn.Module,
             val_loader: DataLoader,
             criterion) -> Tuple[float, float]:
    """
    Smart validation that finds the optimal threshold for the current epoch.
    Returns average loss and best accuracy.
    Args:
        model: The Siamese network model.
        val_loader: DataLoader for validation data.
        criterion: Loss function (e.g., Contrastive Loss).
    Returns:
        avg_loss: Average loss over the validation set.
        best_acc: Best accuracy achieved by scanning thresholds.
    """
    model.eval()
    total_loss = 0.0
    
    all_dists = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            
            # Forward
            v1, v2 = model(img1, img2)
            
            # Loss (Contrastive)
            loss = criterion(v1, v2, labels.squeeze())
            total_loss += loss.item() * img1.size(0)
            
            # Collect distances for accuracy calculation
            # Use Pairwise Euclidean
            dist = torch.nn.functional.pairwise_distance(v1, v2)
            all_dists.extend(dist.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
            
    avg_loss = total_loss / len(all_labels)
    
    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)
    
    best_acc = 0.0
    # Scan min to max distance
    min_d, max_d = all_dists.min(), all_dists.max()
    
    # Check 100 thresholds across the range
    thresholds = np.linspace(min_d, max_d, 100)
    
    for thresh in thresholds:
        predictions = (all_dists < thresh).astype(int)
        acc = (predictions == all_labels).mean()
        if acc > best_acc:
            best_acc = acc
            
    return avg_loss, best_acc


def inverse_transform(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor back to numpy image for plotting.
    Args:
        img_tensor: Image tensor of shape (1, C, H, W).
    Returns:
        img: Numpy array of shape (H, W) or (H, W, C).
    """
    img = img_tensor.squeeze().cpu().numpy()
    return img

def visualize_hardest_errors(fn_pairs: List[Tuple[float, torch.Tensor, torch.Tensor]],
                             fp_pairs: List[Tuple[float, torch.Tensor, torch.Tensor]],
                             n_examples: int = 3,
                             save_path: str = "hardest_errors.png"):
    """
    Visualizes the top N hardest False Negatives and False Positives.
    
    Args:
        fn_pairs: List of tuples (dist, img1, img2) for False Negatives (sorted desc).
        fp_pairs: List of tuples (dist, img1, img2) for False Positives (sorted asc).
        n_examples: Number of examples to show for each type.
        save_path: Path to save the resulting plot.

    """    
    rows = n_examples
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    
    # Ensure axes is always 2D array even if rows=1
    if rows == 1:
        axes = axes.reshape(1, -1)

    cols_titles = ["False Neg (Img A)", "False Neg (Img B)", "False Pos (Img A)", "False Pos (Img B)"]
    for ax, col_title in zip(axes[0], cols_titles):
        ax.set_title(col_title, fontsize=10, fontweight='bold')

    for i in range(rows):
        # False Negatives
        if i < len(fn_pairs):
            dist, img1, img2 = fn_pairs[i]
            
            axes[i, 0].imshow(inverse_transform(img1), cmap='gray')
            axes[i, 0].set_ylabel(f"Pair #{i+1}", fontsize=9)
            
            axes[i, 1].imshow(inverse_transform(img2), cmap='gray')

            axes[i, 1].text(0.5, -0.15, f"Dist: {dist:.4f}\n(Should be Low)", 
                            ha='center', transform=axes[i, 1].transAxes, color='red', fontsize=9)
        else:
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

        # False Positives
        if i < len(fp_pairs):
            dist, img1, img2 = fp_pairs[i]
            
            axes[i, 2].imshow(inverse_transform(img1), cmap='gray')
            
            axes[i, 3].imshow(inverse_transform(img2), cmap='gray')

            axes[i, 3].text(0.5, -0.15, f"Dist: {dist:.4f}\n(Should be High)", 
                            ha='center', transform=axes[i, 3].transAxes, color='red', fontsize=9)
        else:
            axes[i, 2].axis('off')
            axes[i, 3].axis('off')

        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved top-{n_examples} error analysis to {save_path}")
    plt.show()


def plot_distance_distribution(labels: List[int],
                               distances: List[float],
                               threshold: float,
                               save_path: str = "distance_distribution.png"):
    """
    Plots a histogram of distances for positive and negative pairs.
    Args:
        labels: List or array of ground truth labels (1 for same, 0 for different).
        distances: List or array of computed distances.
        threshold: The optimal threshold to mark on the plot.
        save_path: Path to save the resulting plot.
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


def find_optimal_threshold(labels: List[int],
                           distances: List[float]) -> Tuple[float, float]:
    """
    Finds the threshold that maximizes accuracy by scanning the ACTUAL range of distances.
    Args:
        labels: Ground truth labels (1 for same, 0 for different).
        distances: Computed distances between pairs.
    Returns:
        best_acc: Best accuracy achieved.
        best_thresh: Threshold that gives the best accuracy.
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

def evaluate(n_error_examples: int = 3):
    """
    Evaluates the Siamese Network on the LFW test set.
    Args:
        n_error_examples: Number of hardest error examples to visualize.
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
    model = SiameseNetwork(backbone_name=MODEL_NAME, pretrained=False).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Weights loaded successfully.")
    else:
        print(f"Error: {MODEL_PATH} not found!")
        return

    model.eval()

    all_dists = []
    all_labels = []

    print("Calculating distances (Phase 1)...")
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            v1, v2 = model(img1, img2)
            dists = F.pairwise_distance(v1, v2)

            all_dists.extend(dists.cpu().numpy())
            all_labels.extend(labels.numpy().flatten())

    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    best_acc, best_thresh = find_optimal_threshold(all_labels, all_dists)

    print("-" * 30)
    print(f"Best Test Accuracy: {best_acc:.2%} at Threshold: {best_thresh:.2f}")
    fpr, tpr, _ = roc_curve(all_labels, -all_dists)
    print(f"ROC AUC Score: {auc(fpr, tpr):.4f}")
    print("-" * 30)

    plot_distance_distribution(all_labels, all_dists, best_thresh)
    
    print(f"Mining top {n_error_examples} hardest errors...")
    
    fn_errors = []
    fp_errors = []

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            v1, v2 = model(img1, img2)
            dists = F.pairwise_distance(v1, v2)
            
            dists_np = dists.cpu().numpy()
            labels_np = labels.numpy().flatten()
            
            # Identify errors in this batch
            for i in range(len(labels_np)):
                dist = dists_np[i]
                label = labels_np[i]
                
                # False Negative
                # Hardest = Largest Distance
                if label == 1 and dist > best_thresh:
                    fn_errors.append((dist, img1[i].cpu(), img2[i].cpu()))
                
                # False Positive
                # Hardest = Smallest Distance
                elif label == 0 and dist < best_thresh:
                    fp_errors.append((dist, img1[i].cpu(), img2[i].cpu()))

    # Sort to find the "Hardest"
    # FNs: Sort by distance DESCENDING (Higher distance = worse error for same person)
    fn_errors.sort(key=lambda x: x[0], reverse=True)
    
    # FPs: Sort by distance ASCENDING (Lower distance = worse error for diff person)
    fp_errors.sort(key=lambda x: x[0], reverse=False)
    
    top_fns = fn_errors[:n_error_examples]
    top_fps = fp_errors[:n_error_examples]
    
    print(f"Found {len(fn_errors)} total FNs and {len(fp_errors)} total FPs.")
    
    visualize_hardest_errors(top_fns, top_fps, n_examples=n_error_examples)

if __name__ == "__main__":
    evaluate()
