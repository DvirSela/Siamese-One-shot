import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import torchvision.transforms as T
from tqdm import tqdm

from src.config import BATCH_SIZE, SEED, MODEL_PATHS_ENSEMBLE
from src.consts import IMG_DIR, DEVICE, TEST_PAIRS_FILE
from src.dataset import SiameseDataset, parse_lfw_pairs
from src.models.siamese_model import SiameseNetwork
from src.utils import set_seed
from src.transforms import get_transforms


def load_model_architecture(model_path: str):
    """
    Determines architecture from filename and loads weights.
    """
    filename = os.path.basename(model_path).lower()
    if 'clip' in filename:
        model = SiameseNetwork(backbone_name='clip_vit_b32', pretrained=False)
    elif "vit" in filename:
        arch = "vit_b_32" if "b_32" in filename else "vit_b_16"
        model = SiameseNetwork(backbone_name=arch, pretrained=False)
    else:
        model = SiameseNetwork(backbone_name='resnet18', pretrained=False)
        
    print(f"   [+] Loaded {filename} as {model.backbone_name}")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"   [!] Error loading weights: {e}")
        return None, None
        
    model.to(DEVICE)
    model.eval()
    return model, model.backbone_name

def get_inference_transform(backbone_name):
    """
    Returns the correct transform based on architecture requirements.
    """
    if "vit" in str(backbone_name).lower():
        # ViT requires strictly 224x224
        return T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return get_transforms(is_train=False)

def find_optimal_threshold(labels, distances):
    best_acc = 0.0
    best_thresh = 0.0
    
    min_d, max_d = np.min(distances), np.max(distances)
    thresholds = np.linspace(min_d, max_d, 1000)
    
    for thresh in thresholds:
        predictions = (distances < thresh).astype(int)
        acc = (predictions == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_acc, best_thresh

def main():
    set_seed(SEED)
    print("--- Ensemble Evaluation Started ---")
    
    try:
        pairs, labels = parse_lfw_pairs(TEST_PAIRS_FILE, IMG_DIR)
        labels_np = np.array(labels)
    except Exception as e:
        print(f"Error reading data: {e}")
        return
    
    MODEL_A_PATH = MODEL_PATHS_ENSEMBLE[0]
    MODEL_B_PATH = MODEL_PATHS_ENSEMBLE[1]
    model_a, name_a = load_model_architecture(MODEL_A_PATH)
    model_b, name_b = load_model_architecture(MODEL_B_PATH)
    
    if not model_a or not model_b:
        print("Failed to load models. Check paths.")
        return

    dataset_a = SiameseDataset(pairs, labels, transform=get_inference_transform(name_a))
    loader_a = DataLoader(dataset_a, batch_size=BATCH_SIZE, shuffle=False)
    
    dataset_b = SiameseDataset(pairs, labels, transform=get_inference_transform(name_b))
    loader_b = DataLoader(dataset_b, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nComputing distances for Model A ({name_a})...")
    dists_a = []
    
    with torch.no_grad():
        for img1, img2, _ in tqdm(loader_a):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            out = model_a(img1, img2)
            
            if isinstance(out, tuple):
                d = F.pairwise_distance(out[0], out[1])
            else:
                d = 1.0 - out.squeeze()
                
            dists_a.extend(d.cpu().numpy())

    print(f"Computing distances for Model B ({name_b})...")
    dists_b = []
    
    with torch.no_grad():
        for img1, img2, _ in tqdm(loader_b):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            out = model_b(img1, img2)
            
            if isinstance(out, tuple):
                d = F.pairwise_distance(out[0], out[1])
            else:
                d = 1.0 - out.squeeze()
                
            dists_b.extend(d.cpu().numpy())

    dists_a = np.array(dists_a)
    dists_b = np.array(dists_b)

    # Normalize Distances (Min-Max Scaling)
    # This is critical because different models may have different distance scales
    norm_a = (dists_a - dists_a.min()) / (dists_a.max() - dists_a.min())
    norm_b = (dists_b - dists_b.min()) / (dists_b.max() - dists_b.min())

    ensemble_dists = (norm_a + norm_b) / 2.0

    acc_a, _ = find_optimal_threshold(labels_np, dists_a)
    acc_b, _ = find_optimal_threshold(labels_np, dists_b)
    acc_ens, thresh_ens = find_optimal_threshold(labels_np, ensemble_dists)
    
    fpr, tpr, _ = roc_curve(labels_np, -ensemble_dists)
    auc_ens = auc(fpr, tpr)

    print("\n" + "="*40)
    print(f"FINAL RESULTS")
    print("-" * 40)
    print(f"Model A Accuracy: {acc_a:.2%}")
    print(f"Model B Accuracy: {acc_b:.2%}")
    print(f"Ensemble Accuracy: {acc_ens:.2%}")
    print(f"Ensemble AUC:      {auc_ens:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()