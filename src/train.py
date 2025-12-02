import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import os
import copy
from tqdm import tqdm

from src.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, SEED
from src.consts import PAIRS_FILE, IMG_DIR, DEVICE
from src.dataset import SiameseDataset, parse_lfw_pairs, split_pairs_by_identity, generate_new_negatives, get_train_val_datasets
from src.models.simple_model import SimpleSiameseNetwork
from src.training_utils import init_weights, get_optimizer, get_loss_function, get_lr_scheduler, adjust_momentum
from src.utils import set_seed


MAX_EPOCHS = 200            # Paper: "We trained each network for a maximum of 200 epochs"
# Paper: "When validation error did not decrease for 20 epochs, we stopped"
PATIENCE = 20
MOMENTUM_START = 0.5        # Paper: "start at 0.5"
MOMENTUM_END = 0.9          # Standard max momentum (paper calls this mu_j)

def get_transforms(is_train=True):
    """
    Implements the Affine Distortions from the paper.
    ADAPTED FOR 250x250 INPUTS (No Resizing).
    """
    if is_train:

        # CORRECT ORDER (PIL based augmentations first):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            
            # 1. Rotation theta [-10, 10]
            transforms.RandomApply([transforms.RandomAffine(degrees=10)], p=0.5),
            
            # 2. Shear rho [-0.3, 0.3] radians ~= [-17, 17] degrees
            transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=17)], p=0.5),
            
            # 3. Scale s [0.8, 1.2]
            transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))], p=0.5),
            
            # 4. Translation tx, ty [-2, 2] pixels (Approx 2% of 105)
            # For 250x250, 2% is roughly 5 pixels.
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))], p=0.5),
            
            transforms.ToTensor()
        ])
    else:
        # Validation/Test: No Resize, just convert
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])


def validate(model, val_loader, criterion):
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


def main():
    set_seed(SEED)
    print(f"Running on: {DEVICE}")

    # --- 2. DATA PREPARATION ---
    # NOW REPLACED BY SINGLE CALL TO DATASET.PY
    train_dataset, val_dataset = get_train_val_datasets(
        pairs_file=PAIRS_FILE,
        img_dir=IMG_DIR,
        val_size=0.2,
        transform_train=get_transforms(is_train=True),
        transform_val=get_transforms(is_train=False)
    )

    # --- 3. DATA LOADERS ---
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 4. MODEL & OPTIMIZER ---
    model = SimpleSiameseNetwork().to(DEVICE)
    init_weights(model)

    criterion = get_loss_function()
    optimizer = get_optimizer(model, lr=LEARNING_RATE,
                              momentum=MOMENTUM_START, weight_decay=0.0005)
    scheduler = get_lr_scheduler(optimizer)

    writer = SummaryWriter('runs/siamese_experiment')

    # --- 5. TRAINING LOOP ---
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    print(
        f"Starting Training for {MAX_EPOCHS} epochs with patience {PATIENCE}...")

    for epoch in tqdm(range(MAX_EPOCHS)):
        curr_momentum = adjust_momentum(
            optimizer, epoch, MAX_EPOCHS, MOMENTUM_END)

        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for i, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(DEVICE), img2.to(
                DEVICE), labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img1.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            total_samples += img1.size(0)

        avg_train_loss = train_loss / total_samples
        avg_train_acc = train_correct / total_samples

        val_loss, val_acc = validate(model, val_loader, criterion)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', avg_train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Hyperparam/Momentum', curr_momentum, epoch)
        writer.add_scalar('Hyperparam/LR', scheduler.get_last_lr()[0], epoch)

        # print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
        #       f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2%} | "
        #       f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_state, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly Stopping triggered at epoch {epoch+1}.")
                break

    writer.close()
    print("Training Complete.")

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Restored best model weights.")


if __name__ == "__main__":
    main()
