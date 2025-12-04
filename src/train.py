import copy

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import BATCH_SIZE, LEARNING_RATE, SEED, NUM_EPOCHS, PATIENCE, MOMENTUM_START, MOMENTUM_END, WEIGHT_DECAY, THRESHOLD
from src.consts import PAIRS_FILE, IMG_DIR, DEVICE
from src.dataset import get_train_val_datasets
from src.models.siamese_model import SiameseNetwork
from src.models.simple_model import SimpleSiameseNetwork
from src.training_utils import init_weights, get_optimizer, get_loss_function, get_lr_scheduler, adjust_momentum
from src.utils import set_seed
from src.transforms import get_transforms
from src.evaluate import validate


def main():
    set_seed(SEED)
    print(f"Running on: {DEVICE}")
    print(f'training with brach and dropout')

    train_dataset, val_dataset = get_train_val_datasets(
        pairs_file=PAIRS_FILE,
        img_dir=IMG_DIR,
        val_size=0.2,
        transform_train=get_transforms(is_train=True),
        transform_val=get_transforms(is_train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SiameseNetwork().to(DEVICE)
    # model = SimpleSiameseNetwork().to(DEVICE)
    init_weights(model)

    criterion = get_loss_function()
    optimizer = get_optimizer(model, lr=LEARNING_RATE,
                              momentum=MOMENTUM_START, weight_decay=WEIGHT_DECAY)
    scheduler = get_lr_scheduler(optimizer)

    writer = SummaryWriter('runs/siamese_experiment')

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    print(
        f"Starting Training for {NUM_EPOCHS} epochs with patience {PATIENCE}")

    for epoch in tqdm(range(NUM_EPOCHS)):
        curr_momentum = adjust_momentum(optimizer, epoch, NUM_EPOCHS, MOMENTUM_END)

        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for i, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            # 1. Forward Pass (Get Vectors)
            v1, v2 = model(img1, img2)
            
            # 2. Calculate Loss
            # Note: labels need to be squeezed to match shape if necessary
            loss = criterion(v1, v2, labels.squeeze())
            loss.backward()
            optimizer.step()

            # 3. Accumulate Train Loss (CRITICAL STEP)
            train_loss += loss.item() * img1.size(0)

            # 4. Calculate Accuracy (Manual Thresholding)
            # Contrastive loss doesn't output probabilities, so we check distance.
            dist = torch.nn.functional.pairwise_distance(v1, v2)
            
            # If distance < threshold (1.0), we predict "Same" (1)
            # If distance > threshold, we predict "Different" (0)
            predicted = (dist < threshold).float()
            
            train_correct += (predicted == labels.squeeze()).sum().item()
            total_samples += img1.size(0)

        # Calculate Averages
        avg_train_loss = train_loss / total_samples
        avg_train_acc = train_correct / total_samples

        # Validation (Make sure validate() is also updated to use distance!)
        val_loss, val_acc = validate(model, val_loader, criterion)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', avg_train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Hyperparam/Momentum', curr_momentum, epoch)
        writer.add_scalar('Hyperparam/LR', scheduler.get_last_lr()[0], epoch)

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
