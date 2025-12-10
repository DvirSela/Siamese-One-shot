import optuna
import torch
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.consts import PAIRS_FILE, IMG_DIR, DEVICE
from src.config import OPTIMIZER, MODEL_NAME
from src.dataset import get_ohem_dataloaders
from src.models.siamese_model import SiameseNetwork
from src.training_utils import get_optimizer, get_loss_function, get_lr_scheduler, adjust_momentum, ContrastiveLoss, BatchHardTripletLoss
from src.utils import set_seed
from src.transforms import get_transforms
from src.evaluate import validate

def objective(trial):
    # Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    margin = trial.suggest_float("margin", 0.5, 2.0, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # Sampler Strategy (Total Batch Size 32)
    sampler_mode = trial.suggest_categorical("sampler_mode", ["8x4", "16x2"])
    
    if sampler_mode == "8x4":
        p_people, k_images = 8, 4
    else:
        p_people, k_images = 16, 2
        
    # Setup Data & Model
    train_dataset, train_sampler, val_dataset = get_ohem_dataloaders(
        PAIRS_FILE, IMG_DIR,
        val_size=0.2,
        transform_train=get_transforms(is_train=True),
        transform_val=get_transforms(is_train=False),
        p=p_people, k=k_images
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Use MODEL_NAME from config so it supports ViT or ResNet dynamically
    model = SiameseNetwork(backbone_name=MODEL_NAME, pretrained=True).to(DEVICE)
    
    criterion_train = BatchHardTripletLoss(margin=margin)
    criterion_val = ContrastiveLoss(margin=margin) 
    
    optimizer = get_optimizer(model, lr=lr, momentum=0.9, weight_decay=weight_decay, optimizer_name=OPTIMIZER)
    scheduler = get_lr_scheduler(optimizer)
    
    MAX_TUNE_EPOCHS = 100
    best_val_acc = 0.0
    
    for epoch in range(MAX_TUNE_EPOCHS):
        curr_momentum = adjust_momentum(optimizer, epoch, MAX_TUNE_EPOCHS, 0.9)
        model.train()
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            embeddings = model.forward_once(images)
            loss = criterion_train(embeddings, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion=criterion_val)
        
        # Step LR
        scheduler.step()
        
        # Track Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        # Optuna Pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val_acc

if __name__ == "__main__":
    set_seed(42)
    print(f"Starting Hyperparameter Tuning for {MODEL_NAME}...")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    
    N_TRIALS = 20

    with tqdm(total=N_TRIALS, desc="Hyperparameter Tuning") as pbar:
        def callback(study, trial):
            pbar.update(1)
            if study.best_value:
                pbar.set_postfix({"Best Acc": f"{study.best_value:.4f}"})

        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

    print("\n----------------------------------")
    print("Tuning Complete.")
    print("Best Validation Accuracy:", study.best_value)
    print("Best Hyperparameters:", study.best_params)
    print("----------------------------------")