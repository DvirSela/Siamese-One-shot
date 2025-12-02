import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, SEED
from src.consts import PAIRS_FILE, IMG_DIR, DEVICE
from src.dataset import SiameseDataset
from src.models.simple_model import SimpleSiameseNetwork
from src.training_utils import get_optimizer, get_loss_function
from src.utils import set_seed


def main():
    set_seed(SEED)
    print(f"Running on: {DEVICE}")

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])

    print("Initializing Dataset...")
    train_dataset = SiameseDataset(PAIRS_FILE, IMG_DIR, transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    data_iter = iter(train_loader)
    try:
        img1_batch, img2_batch, label_batch = next(data_iter)
    except StopIteration:
        print("Error: Dataset is empty. Check paths.")
        return

    img1_batch = img1_batch.to(DEVICE)
    img2_batch = img2_batch.to(DEVICE)
    label_batch = label_batch.to(DEVICE).unsqueeze(1)  # Shape (N, 1)

    print(f"Batch Shapes - Img: {img1_batch.shape}, Label: {label_batch.shape}")
    print("Labels in batch:", label_batch.flatten().tolist())

    model = SimpleSiameseNetwork().to(DEVICE)

    optimizer = get_optimizer(model, lr=LEARNING_RATE)
    criterion = get_loss_function()

    writer = SummaryWriter('runs/overfit_sanity_check')

    print("Starting Sanity Check (Overfitting on ONE batch)...")
    for epoch in range(NUM_EPOCHS):
        model.train()

        optimizer.zero_grad()

        outputs = model(img1_batch, img2_batch)

        loss = criterion(outputs, label_batch)

        loss.backward()
        optimizer.step()

        predicted = (outputs > 0.5).float()
        accuracy = (predicted == label_batch).float().mean()

        # Logging
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Accuracy/train', accuracy.item(), epoch)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Loss: {loss.item():.4f} Acc: {accuracy.item():.4f}")

        if accuracy.item() == 1.0 and loss.item() < 0.005:
            print(f"SUCCESS! Overfitted batch at epoch {epoch}")
            break

    writer.close()
    print("Sanity Check Complete: Loss -> 0 <-> pipeline works.")

from src.dataset import parse_lfw_pairs, split_pairs_by_identity,generate_new_negatives
import os
if __name__ == "__main__":
    # Load all training data
    all_train_pairs, all_train_labels = parse_lfw_pairs(PAIRS_FILE, IMG_DIR)

    (train_pairs, train_labels), (val_pairs, val_labels) = split_pairs_by_identity(
        all_train_pairs, all_train_labels, val_size=0.2
    )

    # 2. Check balance
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    print(f"Original Val Balance: {val_pos} Pos, {val_neg} Neg")
    # 3. Fix Balance if needed
    if val_pos > val_neg:
        needed = val_pos - val_neg
        print(f"Generating {needed} new negative pairs to balance validation...")
        
        # Extract just the names in the validation set to ensure no leakage
        # (We re-derive valid names from the surviving val_pairs)
        val_names_set = set()
        for p1, p2 in val_pairs:
            val_names_set.add(p1.split(os.sep)[-2])
            val_names_set.add(p2.split(os.sep)[-2])
            
        new_p, new_l = generate_new_negatives(list(val_names_set), needed, IMG_DIR)
        
        val_pairs.extend(new_p)
        val_labels.extend(new_l)

    print(f"Final Val Count: {len(val_labels)} ({sum(val_labels)} Pos, {len(val_labels)-sum(val_labels)} Neg)")