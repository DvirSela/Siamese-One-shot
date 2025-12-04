import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import BATCH_SIZE, LEARNING_RATE, SEED, NUM_EPOCHS, PATIENCE, MOMENTUM_START, MOMENTUM_END, WEIGHT_DECAY, MODEL_NAME
from src.consts import PAIRS_FILE, IMG_DIR, DEVICE

from src.dataset import get_dataloaders 
from src.models.siamese_model import SiameseNetwork
from src.training_utils import get_optimizer, get_loss_function, get_lr_scheduler, adjust_momentum, ContrastiveLoss
from src.utils import set_seed
from src.transforms import get_transforms
from src.evaluate import validate

def main():
    set_seed(SEED)
    print(f"Running on: {DEVICE}")
    print(f'Training with Triplet Loss')

    # Single call to get both datasets safely
    train_dataset, val_dataset = get_dataloaders(
        PAIRS_FILE, IMG_DIR,
        val_size=0.2,
        transform_train=get_transforms(is_train=True),
        transform_val=get_transforms(is_train=False),
        use_triplet=True # Enable Triplet Mode for Training
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Initializing ResNet Backbone...")
    model = SiameseNetwork(backbone_name=MODEL_NAME, pretrained=True).to(DEVICE)

    # Triplet Loss for Training
    criterion_train = get_loss_function(type='triplet')
    
    # Contrastive Loss for Validation (since Val is pairs)
    criterion_val = ContrastiveLoss(margin=1.0) 

    optimizer = get_optimizer(model, lr=LEARNING_RATE,
                              momentum=MOMENTUM_START, weight_decay=WEIGHT_DECAY)
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

        # TRIPLET LOOP
        for i, (anchor, pos, neg) in enumerate(train_loader):
            anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward 3 times
            v_a = model.forward_once(anchor)
            v_p = model.forward_once(pos)
            v_n = model.forward_once(neg)
            
            # Calculate Triplet Loss
            loss = criterion_train(v_a, v_p, v_n)
            
            loss.backward()
            optimizer.step()
            
            # Accumulate Loss
            train_loss += loss.item() * anchor.size(0)
            total_samples += anchor.size(0)

        avg_train_loss = train_loss / total_samples
        
        # Validation (Pairwise Metric)
        # We pass criterion_val (Contrastive) because validation data is pairs
        val_loss, val_acc = validate(model, val_loader, criterion=criterion_val)        
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch) # Train Acc is hard to define for triplets
        writer.add_scalar('Hyperparam/Momentum', curr_momentum, epoch)
        writer.add_scalar('Hyperparam/LR', scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

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