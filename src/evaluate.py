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
import torchvision.transforms as T

from src.models.simple_model import SimpleSiameseNetwork
from src.config import BATCH_SIZE, SEED, MODEL_NAME
from src.consts import IMG_DIR, DEVICE, TEST_PAIRS_FILE, MODEL_PATH
from src.dataset import SiameseDataset, parse_lfw_pairs
from src.models.siamese_model import SiameseNetwork
from src.utils import set_seed
from src.transforms import get_transforms


def inverse_transform(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.squeeze().cpu().numpy()
    return img


def save_confusion_matrix(cm: np.ndarray, model_name: str, save_dir: str):
    """
    Saves the confusion matrix as an image file.
    Args:
        cm (np.ndarray): Confusion matrix.
        model_name (str): Name of the model (for title and filename).
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


def plot_distance_distribution(labels: List[int], distances: List[float], threshold: float, model_name: str, save_dir: str):
    """
    Plots the distribution of distances for positive and negative pairs.
    Args:
        labels (List[int]): List of ground truth labels (1 for same, 0 for different).
        distances (List[float]): List of computed distances.
        threshold (float): Decision threshold.
        model_name (str): Name of the model (for title and filename).
        save_dir (str): Directory to save the image.
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
    plt.title(f'Distribution: {model_name}')
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
                             model_name: str,
                             save_dir: str,
                             n_examples: int = 3):
    """
    Visualizes the hardest false negatives and false positives.
    Args:
        fn_pairs (List[Tuple[float, torch.Tensor, torch.Tensor]]): List of false negative pairs (distance, img1, img2).
        fp_pairs (List[Tuple[float, torch.Tensor, torch.Tensor]]): List of false positive pairs (distance, img1, img2).
        model_name (str): Name of the model (for title and filename).
        save_dir (str): Directory to save the image.
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
    Attempts to load a model checkpoint by trying multiple architectures.
    Args:
        model_path (str): Path to the model checkpoint.
    Returns:
        Optional[torch.nn.Module]: Loaded model or None if failed.
    """
    filename = os.path.basename(model_path).lower()
    candidates = []

    if "vit" in filename:
        candidates.append(("ViT_B_16", SiameseNetwork(
            backbone_name='vit_b_16', pretrained=False)))
    elif "resnet" in filename or "tuned" in filename or "triplet" in filename:
        candidates.append(("ResNet18", SiameseNetwork(
            backbone_name='resnet18', pretrained=False)))
    elif "batchnorm" in filename or "vanilla" in filename:
        if SimpleSiameseNetwork:
            candidates.append(("CustomCNN", SimpleSiameseNetwork()))

    candidates.append(("ResNet18", SiameseNetwork(
        backbone_name='resnet18', pretrained=False)))
    if SimpleSiameseNetwork:
        candidates.append(("CustomCNN", SimpleSiameseNetwork()))
    candidates.append(("ViT_B_16", SiameseNetwork(
        backbone_name='vit_b_16', pretrained=False)))

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    for arch_name, model in candidates:
        try:
            model.to(DEVICE)
            model.load_state_dict(checkpoint)
            return model
        except RuntimeError:
            continue
        except Exception as e:
            print(f"   -> Error checking {arch_name}: {e}")
            continue
    print(f"   -> FAILED to match architecture for {filename}")
    return None


def find_optimal_threshold(labels: List[int], distances: List[float]) -> Tuple[float, float]:
    """
    Finds the optimal threshold that maximizes accuracy.
    Args:
        labels (List[int]): Ground truth labels.
        distances (List[float]): Computed distances.
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


def evaluate_model(model_path: str, n_error_examples: int = 3, visualize: bool = True) -> Optional[Dict]:
    """
    Evaluates a single model on the LFW dataset.
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
        test_pairs, test_labels = parse_lfw_pairs(TEST_PAIRS_FILE, IMG_DIR)
    except Exception as e:
        print(f"   [!] Error parsing pairs: {e}")
        return None

    # 1. Load Model
    model = load_model_safely(model_path)
    if model is None:
        return None

    # 2. Determine Transform dynamically based on Architecture
    backbone = getattr(model, 'backbone_name', '').lower()

    if 'vit' in backbone or 'vit' in model_name.lower():
        # ViT REQUIRES 224x224
        test_transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        # ResNet / Custom (Use default 250x250 or config default)
        test_transform = get_transforms(is_train=False)

    test_dataset = SiameseDataset(
        test_pairs, test_labels, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_dists = []
    all_labels = []

    fn_candidates = []
    fp_candidates = []

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            output = model(img1, img2)

            if isinstance(output, tuple):
                v1, v2 = output
                dists = F.pairwise_distance(v1, v2)
            else:
                dists = 1.0 - output.squeeze()

            batch_dists = dists.cpu().numpy()
            batch_labels = labels.numpy().flatten()

            all_dists.extend(batch_dists)
            all_labels.extend(batch_labels)

            if visualize:
                for i in range(len(batch_labels)):
                    fn_candidates.append(
                        (batch_dists[i], img1[i].cpu(), img2[i].cpu(), batch_labels[i]))
                    fp_candidates.append(
                        (batch_dists[i], img1[i].cpu(), img2[i].cpu(), batch_labels[i]))

    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)
    best_acc, best_thresh = find_optimal_threshold(all_labels, all_dists)
    fpr, tpr, _ = roc_curve(all_labels, -all_dists)
    roc_auc_score = auc(fpr, tpr)

    final_preds = (all_dists < best_thresh).astype(int)
    cm = confusion_matrix(all_labels, final_preds)
    tn, fp, fn, tp = cm.ravel()

    print(
        f"   -> Acc: {best_acc:.2%} | AUC: {roc_auc_score:.4f} | Thresh: {best_thresh:.4f}")
    print(f"   -> CM: TP:{tp} FN:{fn} FP:{fp} TN:{tn}")

    if visualize:
        save_confusion_matrix(cm, model_name, images_dir)
        plot_distance_distribution(
            all_labels, all_dists, best_thresh, model_name, images_dir)

        real_fns = []
        real_fps = []
        for dist, i1, i2, lbl in fn_candidates:
            if lbl == 1 and dist > best_thresh:
                real_fns.append((dist, i1, i2))
        for dist, i1, i2, lbl in fp_candidates:
            if lbl == 0 and dist < best_thresh:
                real_fps.append((dist, i1, i2))

        real_fns.sort(key=lambda x: x[0], reverse=True)
        real_fps.sort(key=lambda x: x[0], reverse=False)
        visualize_hardest_errors(
            real_fns[:n_error_examples], real_fps[:n_error_examples], model_name, images_dir, 3)

    return {"model": model_name, "accuracy": best_acc, "auc": roc_auc_score, "threshold": best_thresh}


def evaluate_all(models_dir: str):
    """Evaluates all models in the specified directory."""
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
    print(f"{'Model Name':<35} | {'Acc':<8} | {'AUC':<8} | {'Thresh':<8}")
    print("-" * 65)
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    for r in results:
        print(
            f"{r['model']:<35} | {r['accuracy']:.2%} | {r['auc']:.4f}   | {r['threshold']:.2f}")
    print("="*65)


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
