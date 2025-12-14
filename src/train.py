import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import LEARNING_RATE, SEED, NUM_EPOCHS, PATIENCE, MOMENTUM_START, MOMENTUM_END, WEIGHT_DECAY, MODEL_NAME, P_PEOPLE, K_IMAGES, OPTIMIZER
from src.consts import PAIRS_FILE, IMG_DIR, DEVICE

from src.dataset import get_dataloaders, get_ohem_dataloaders
from src.models.siamese_model import SiameseNetwork
from src.training_utils import get_optimizer, get_loss_function, get_lr_scheduler, adjust_momentum, ContrastiveLoss
from src.utils import set_seed
from src.transforms import get_transforms
from src.evaluate import validate

def main():
    set_seed(SEED)
    print(f"Running on: {DEVICE}")
    print(f'Training with OHEM (Batch Hard)')

    train_dataset, train_sampler, val_dataset = get_ohem_dataloaders(
        PAIRS_FILE, IMG_DIR,
        val_size=0.2,
        transform_train=get_transforms(is_train=True),
        transform_val=get_transforms(is_train=False),
        p=P_PEOPLE, k=K_IMAGES
    )

    # Train Loader needs the SAMPLER (Shuffle must be False when using Sampler)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("Initializing Backbone...")
    model = SiameseNetwork(backbone_name=MODEL_NAME, pretrained=True).to(DEVICE)

    # OHEM Loss
    criterion_train = get_loss_function(loss_type='ohem')
    # Pairwise Validation Loss
    criterion_val = ContrastiveLoss(margin=1.0) 

    optimizer = get_optimizer(model, lr=LEARNING_RATE, momentum=MOMENTUM_START, weight_decay=WEIGHT_DECAY, optimizer_name=OPTIMIZER)
    scheduler = get_lr_scheduler(optimizer)

    writer = SummaryWriter('runs/siamese_experiment')

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    print(f"Starting Training for {NUM_EPOCHS} epochs with patience {PATIENCE}")

    for epoch in tqdm(range(NUM_EPOCHS)):
        curr_momentum = adjust_momentum(optimizer, epoch, NUM_EPOCHS, MOMENTUM_END)

        model.train()
        train_loss = 0.0
        total_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward once (Get embeddings for whole batch)
            embeddings = model.forward_once(images)
            
            # Loss handles the mining internally
            loss = criterion_train(embeddings, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_train_loss = train_loss / total_samples

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion=criterion_val)        
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch) 
        writer.add_scalar('Hyperparam/Momentum', curr_momentum, epoch)
        writer.add_scalar('Hyperparam/LR', scheduler.get_last_lr()[0], epoch)
        
        print(f"T Loss: {avg_train_loss:.4f} | V Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

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