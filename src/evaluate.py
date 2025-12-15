import os
import glob
import argparse

from typing import Tuple, List, Dict, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

from src.models.simple_model import SimpleSiameseNetwork

from src.config import BATCH_SIZE, SEED, MODEL_NAME
# Added PAIRS_FILE
from src.consts import IMG_DIR, DEVICE, TEST_PAIRS_FILE, MODEL_PATH, PAIRS_FILE
from src.dataset import SiameseDataset, parse_lfw_pairs
from src.models.siamese_model import SiameseNetwork
from src.utils import set_seed
from src.transforms import get_transforms


def inverse_transform(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Inverse the normalization for visualization.
    Args:
        img_tensor (torch.Tensor): Normalized image tensor.
    Returns:
        np.ndarray: Denormalized image as a NumPy array.
    """
    img = img_tensor.squeeze().cpu().numpy()
    return img


def save_confusion_matrix(cm: np.ndarray, model_name: str, save_dir: str):
    """
    Save confusion matrix as an image.
    Args:
        cm (np.ndarray): Confusion matrix.
        model_name (str): Name of the model.
        save_dir (str): Directory to save the image.
    """
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Diff", "Same"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix: {model_name}")
    save_path = os.path.join(save_dir, f"cm_{model_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   [+] Saved Confusion Matrix to {save_path}")


def plot_distance_distribution(labels: np.ndarray, distances: np.ndarray, threshold: float, model_name: str, save_dir: str):
    """
    Plot and save the distribution of distances.
    Args:
        labels (np.ndarray): Ground truth labels.
        distances (np.ndarray): Computed distances.
        threshold (float): Decision threshold.
        model_name (str): Name of the model.
        save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    labels = np.array(labels)
    distances = np.array(distances)
    pos_dists = distances[labels == 1]
    neg_dists = distances[labels == 0]
    plt.hist(pos_dists, bins=50, alpha=0.5,
             label='Positive (Same)', color='green')
    plt.hist(neg_dists, bins=50, alpha=0.5,
             label='Negative (Diff)', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--',
                label=f'Threshold ({threshold:.2f})')
    plt.title(f'Distribution: {model_name} (Test Set)')
    plt.xlabel('Distance (Lower = Same)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f"dist_{model_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   [+] Saved Distance Plot to {save_path}")


def visualize_hardest_errors(fn_pairs: List[Tuple[float, torch.Tensor, torch.Tensor]],
                             fp_pairs: List[Tuple[float, torch.Tensor, torch.Tensor]],
                             model_name: str, save_dir: str, n_examples: int = 3):
    """
    Visualize hardest false negatives and false positives.
    Args:
        fn_pairs (List[Tuple[float, torch.Tensor, torch.Tensor]]): List of false negative pairs (distance, img1, img2).
        fp_pairs (List[Tuple[float, torch.Tensor, torch.Tensor]]): List of false positive pairs (distance, img1, img2).
        model_name (str): Name of the model.
        save_dir (str): Directory to save the visualization.
        n_examples (int): Number of examples to visualize for each error type.
    """

    rows = n_examples
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    cols_titles = ["False Neg (Img A)", "False Neg (Img B)",
                   "False Pos (Img A)", "False Pos (Img B)"]
    for ax, col_title in zip(axes[0], cols_titles):
        ax.set_title(col_title, fontsize=10, fontweight='bold')
    for i in range(rows):
        if i < len(fn_pairs):
            dist, img1, img2 = fn_pairs[i]
            axes[i, 0].imshow(inverse_transform(img1), cmap='gray')
            axes[i, 0].set_ylabel(f"FN #{i+1}", fontsize=9)
            axes[i, 1].imshow(inverse_transform(img2), cmap='gray')
            axes[i, 1].text(0.5, -0.15, f"Dist: {dist:.4f}\n(High)", ha='center',
                            transform=axes[i, 1].transAxes, color='red', fontsize=9)
        else:
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
        if i < len(fp_pairs):
            dist, img1, img2 = fp_pairs[i]
            axes[i, 2].imshow(inverse_transform(img1), cmap='gray')
            axes[i, 3].imshow(inverse_transform(img2), cmap='gray')
            axes[i, 3].text(0.5, -0.15, f"Dist: {dist:.4f}\n(Low)", ha='center',
                            transform=axes[i, 3].transAxes, color='red', fontsize=9)
        else:
            axes[i, 2].axis('off')
            axes[i, 3].axis('off')
        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"errors_{model_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   [+] Saved Error Analysis to {save_path}")


def load_model_safely(model_path: str) -> Optional[torch.nn.Module]:
    """
    Load a model checkpoint by trying multiple architectures.
    Args:
        model_path (str): Path to the model checkpoint.
    Returns:
        Optional[torch.nn.Module]: Loaded model or None if failed.
    """
    filename = os.path.basename(model_path).lower()
    candidates = []

    if "best_model" in filename and "vit" not in filename:
        candidates.append(("ViT_B_32", SiameseNetwork(
            backbone_name='vit_b_32', pretrained=False)))
    if "clip" in filename:
        candidates.append(("CLIP_ViT_B32", SiameseNetwork(
            backbone_name='clip_vit_b32', pretrained=False)))
    if "vit" in filename:
        candidates.append(("ViT_B_32", SiameseNetwork(
            backbone_name='vit_b_32', pretrained=False)))
    elif "resnet" in filename or "tuned" in filename or "triplet" in filename:
        candidates.append(("ResNet18", SiameseNetwork(
            backbone_name='resnet18', pretrained=False)))
    elif "batchnorm" in filename or "vanilla" in filename:
        if SimpleSiameseNetwork:
            candidates.append(("CustomCNN", SimpleSiameseNetwork()))

    candidates.append(("ResNet18", SiameseNetwork(
        backbone_name='resnet18', pretrained=False)))
    candidates.append(("ViT_B_32", SiameseNetwork(
        backbone_name='vit_b_32', pretrained=False)))
    if SimpleSiameseNetwork:
        candidates.append(("CustomCNN", SimpleSiameseNetwork()))

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    for arch_name, model in candidates:
        try:
            model.to(DEVICE)
            model.load_state_dict(checkpoint)
            return model
        except RuntimeError:
            continue
        except Exception as e:
            continue
    print(f"   -> FAILED to match architecture for {filename}")
    return None


def find_optimal_threshold(labels: np.ndarray, distances: np.ndarray) -> Tuple[float, float]:
    """
    Find the optimal threshold that maximizes accuracy.
    Args:
        labels (np.ndarray): Ground truth labels.
        distances (np.ndarray): Computed distances.
    Returns:
        Tuple[float, float]: Best accuracy and corresponding threshold.
    """
    best_acc = 0.0
    best_thresh = 0.0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    thresholds = np.linspace(min_dist, max_dist, num=1000)
    for thresh in thresholds:
        predictions = (distances < thresh).astype(int)
        accuracy = (predictions == labels).mean()
        if accuracy > best_acc:
            best_acc = accuracy
            best_thresh = thresh
    return best_acc, best_thresh


def get_distances_and_labels(model: torch.nn.Module,
                             loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Compute distances and collect labels from the DataLoader.
    Args:
        model (torch.nn.Module): The Siamese network model.
        loader (DataLoader): DataLoader for the dataset.
    Returns:
        Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]: Distances, labels, img1 tensors, img2 tensors.
    """
    all_dists = []
    all_labels = []
    all_img1 = []
    all_img2 = []

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            output = model(img1, img2)

            if isinstance(output, tuple):
                dists = F.pairwise_distance(output[0], output[1])
            else:
                dists = 1.0 - output.squeeze()

            all_dists.extend(dists.cpu().numpy())
            all_labels.extend(labels.numpy().flatten())
            all_img1.append(img1.cpu())
            all_img2.append(img2.cpu())

    return np.array(all_dists), np.array(all_labels), torch.cat(all_img1), torch.cat(all_img2)


def evaluate_model(model_path: str, n_error_examples: int = 3, visualize: bool = True) -> Optional[Dict]:
    """
    Evaluate a single model on the LFW dataset.
    Args:
        model_path (str): Path to the model checkpoint.
        n_error_examples (int): Number of error examples to visualize.
        visualize (bool): Whether to generate visualizations.
    Returns:
        Optional[Dict]: Evaluation results or None if failed.
    """
    set_seed(SEED)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\nEvaluating: {model_name} ...")

    images_dir = "./images"
    os.makedirs(images_dir, exist_ok=True)

    try:
        train_val_pairs, train_val_labels = parse_lfw_pairs(
            PAIRS_FILE, IMG_DIR)

        # Reproduce the exact split used in training
        _, val_pairs, _, val_labels = train_test_split(
            train_val_pairs, train_val_labels, test_size=0.2, random_state=SEED, stratify=train_val_labels
        )
        print(
            f"   -> Loaded {len(val_pairs)} Validation pairs for Calibration.")
    except Exception as e:
        print(f"   [!] Error loading validation pairs: {e}")
        return None

    try:
        test_pairs, test_labels = parse_lfw_pairs(TEST_PAIRS_FILE, IMG_DIR)
        print(
            f"   -> Loaded {len(test_pairs)} Test pairs for Blind Evaluation.")
    except Exception as e:
        print(f"   [!] Error loading test pairs: {e}")
        return None

    model = load_model_safely(model_path)
    if model is None:
        return None

    backbone = getattr(model, 'backbone_name', '').lower()
    is_clip = 'clip' in backbone

    # validation transforms (no aug) for both sets
    test_transform = get_transforms(is_train=False, use_clip=is_clip)

    val_dataset = SiameseDataset(
        val_pairs, val_labels, transform=test_transform)
    test_dataset = SiameseDataset(
        test_pairs, test_labels, transform=test_transform)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()

    # 4. Calibration (Use Validation Split)
    # We find the best threshold on the validation set, which the model has "seen"
    # indirectly (via hyperparam tuning), but hasn't trained on.
    val_dists, val_lbls, _, _ = get_distances_and_labels(model, val_loader)
    _, calibrated_threshold = find_optimal_threshold(val_lbls, val_dists)
    print(
        f"   -> Calibrated Threshold (from Val Set): {calibrated_threshold:.4f}")

    test_dists, test_lbls, test_img1, test_img2 = get_distances_and_labels(
        model, test_loader)

    # Accuracy using the FIXED calibrated threshold
    final_preds = (test_dists < calibrated_threshold).astype(int)
    blind_acc = (final_preds == test_lbls).mean()

    fpr, tpr, _ = roc_curve(test_lbls, -test_dists)
    roc_auc_score = auc(fpr, tpr)

    cm = confusion_matrix(test_lbls, final_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"   -> Blind Test Acc: {blind_acc:.2%} | AUC: {roc_auc_score:.4f}")
    print(f"   -> CM: TP:{tp} FN:{fn} FP:{fp} TN:{tn}")

    if visualize:
        save_confusion_matrix(cm, model_name, images_dir)
        plot_distance_distribution(
            test_lbls, test_dists, calibrated_threshold, model_name, images_dir)

        fn_candidates = []
        fp_candidates = []

        for i in range(len(test_lbls)):
            dist = test_dists[i]
            label = test_lbls[i]

            if label == 1 and dist > calibrated_threshold:
                fn_candidates.append((dist, test_img1[i], test_img2[i]))
            elif label == 0 and dist < calibrated_threshold:
                fp_candidates.append((dist, test_img1[i], test_img2[i]))

        fn_candidates.sort(key=lambda x: x[0], reverse=True)
        fp_candidates.sort(key=lambda x: x[0], reverse=False)

        visualize_hardest_errors(fn_candidates[:n_error_examples],
                                 fp_candidates[:n_error_examples],
                                 model_name, images_dir, 3)

    return {"model": model_name, "accuracy": blind_acc, "auc": roc_auc_score, "threshold": calibrated_threshold}


def evaluate_all(models_dir: str):
    """
    Evaluate all .pth models in the specified directory.
    Args:
        models_dir (str): Directory containing model checkpoints.
    """
    print(f"Scanning {models_dir} for .pth files...")
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    if not model_files:
        print("No models found.")
        return
    results = []
    for model_path in model_files:
        res = evaluate_model(model_path, visualize=True)
        if res:
            results.append(res)
    print("\n" + "="*65)
    print(f"{'Model Name':<35} | {'Blind Acc':<10} | {'AUC':<8} | {'Thresh':<8}")
    print("-" * 65)
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    for r in results:
        print(
            f"{r['model']:<35} | {r['accuracy']:.2%}    | {r['auc']:.4f}   | {r['threshold']:.2f}")
    print("="*65)


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
            img1, img2, labels = img1.to(DEVICE), img2.to(
                DEVICE), labels.to(DEVICE)

            v1, v2 = model(img1, img2)

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

    min_d, max_d = all_dists.min(), all_dists.max()

    thresholds = np.linspace(min_d, max_d, 100)

    for thresh in thresholds:
        predictions = (all_dists < thresh).astype(int)
        acc = (predictions == all_labels).mean()
        if acc > best_acc:
            best_acc = acc

    return avg_loss, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=['single', 'all'], default='single')
    parser.add_argument("--dir", type=str, default="./")
    args = parser.parse_args()
    if args.mode == "single":
        evaluate_model(MODEL_PATH, visualize=True)
    else:
        evaluate_all(models_dir=args.dir)
