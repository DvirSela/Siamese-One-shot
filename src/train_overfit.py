import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


from src.utils import set_seed
from src.dataset import SiameseDataset
from src.models.simple_model import SimpleSiameseNetwork
from src.consts import PAIRS_FILE, IMG_DIR, DEVICE
from src.training_utils import get_optimizer, get_loss_function
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 100


def main():
    set_seed(42)
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
    label_batch = label_batch.to(DEVICE).unsqueeze(1) # Shape (N, 1)

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

if __name__ == "__main__":
    main()