import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torch

from src.config import MARGIN


def init_weights(model):
    """
    Applies the specific weight initialization from the Siamese Paper.
    Conv Layers: Weights ~ N(0, 1e-2), Biases ~ N(0.5, 1e-2)
    FC Layers: Weights ~ N(0, 2e-1), Biases ~ N(0.5, 1e-2)
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Conv: Weights N(0, 1e-2), Biases N(0.5, 1e-2)
            nn.init.normal_(m.weight, 0, 1e-2)
            if m.bias is not None:
                nn.init.normal_(m.bias, 0.5, 1e-2)

        elif isinstance(m, nn.Linear):
            # FC: Weights N(0, 2e-1), Biases N(0.5, 1e-2)
            nn.init.normal_(m.weight, 0, 2e-1)
            if m.bias is not None:
                nn.init.normal_(m.bias, 0.5, 1e-2)

    print("Weights initialized according to paper specifications.")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Contrastive Loss Formula:
        # If Label=1 (Same): Loss = distance^2
        # If Label=0 (Diff): Loss = max(0, margin - distance)^2
        # Note: We assume label is 1 for Same, 0 for Diff.

        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin -
                                                euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Distance(A, P)
        d_pos = F.pairwise_distance(anchor, positive)
        # Distance(A, N)
        d_neg = F.pairwise_distance(anchor, negative)

        # Loss = max(0, D_pos - D_neg + margin)
        losses = torch.relu(d_pos - d_neg + self.margin)

        return losses.mean()


class TripletCosineLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Cosine Similarity is between -1 and 1.
        # We want "Cosine Distance" where 0 is same, 2 is opposite.
        # Dist = 1 - Similarity

        d_pos = 1 - F.cosine_similarity(anchor, positive, dim=1)
        d_neg = 1 - F.cosine_similarity(anchor, negative, dim=1)

        # Loss = max(0, D_pos - D_neg + margin)
        losses = torch.relu(d_pos - d_neg + self.margin)

        return losses.mean()
class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: (Batch, Embed_Dim)
        labels: (Batch,)
        """
        # 1. Compute Pairwise Euclidean Distance Matrix
        # Formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        
        dot_product = torch.mm(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        # Expand dimensions to create the distance matrix
        # (N, 1) - (N, N) + (1, N) broadcasting
        dist_sq = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        
        # Numerical stability (ensure non-negative)
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        # Actual Distance (Sqrt) + epsilon for gradients at 0
        dist_matrix = torch.sqrt(dist_sq + 1e-8)
        
        # 2. Hardest Positive (Max Distance)
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.t()).bool()
        
        # Fill non-positive entries with 0.0 so they aren't picked as max
        hardest_pos_dist = (dist_matrix * mask_pos.float()).max(dim=1)[0]
        
        # 3. Hardest Negative (Min Distance)
        mask_neg = (labels != labels.t()).bool()
        
        # Fill non-negative entries with a large value so they aren't picked as min
        max_dist_val = torch.max(dist_matrix) + 10.0
        hardest_neg_dist = (dist_matrix + (~mask_neg).float() * max_dist_val).min(dim=1)[0]
        
        # 4. Compute Loss
        # Loss = mean( ReLU(Hardest_Pos - Hardest_Neg + Margin) )
        loss = torch.mean(torch.relu(hardest_pos_dist - hardest_neg_dist + self.margin))
        
        return loss


def get_loss_function(loss_type='triplet_cosine') -> nn.Module:
    """
    Returns the specified loss function.
    Args:
        loss_type (str): 'triplet', 'contrastive', 'triplet_cosine'
    Raises:
        ValueError: If loss_type is not recognized.
    Returns:
        nn.Module: Loss function instance.
    """
    if loss_type == 'triplet':
        return TripletLoss(margin=1.0)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(margin=1.0)
    elif loss_type == 'triplet_cosine':
        return TripletCosineLoss(margin=MARGIN)
    elif loss_type == 'ohem':
        return BatchHardTripletLoss(margin=MARGIN)
    else:
        raise ValueError(f"Loss type {loss_type} not recognized.")


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float = 0.01, momentum: float = 0.5, weight_decay: float = 1e-4) -> optim.Optimizer:
    """
    Paper: SGD with momentum and L2 regularization.

    Args:
        model: PyTorch model
        lr: learning rate
        optimizer: Optimizer type. A string, e.g., 'SGD'
        momentum: Initial momentum (paper starts at 0.5)
        weight_decay: This is the lambda parameter from the paper's loss equation.
    Returns:
        optimizer: Configured SGD optimizer
    """
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
    return optimizer


def get_lr_scheduler(optimizer):
    """
    Paper: 'decayed uniformly across the network by 1 percent per epoch'
    Formula: lr_new = lr_old * 0.99
    """
    return ExponentialLR(optimizer, gamma=0.99)


def adjust_momentum(optimizer, epoch, max_epochs, target_momentum=0.9):
    """
    Paper: 'fixed momentum to start at 0.5... increasing linearly each epoch'

    This function should be called inside the training loop at the start of each epoch.
    It linearly interpolates momentum from 0.5 to target_momentum.
    """

    start_momentum = 0.5

    if epoch < max_epochs:
        # Linear interpolation formula
        new_momentum = start_momentum + \
            (epoch / max_epochs) * (target_momentum - start_momentum)
    else:
        new_momentum = target_momentum

    # Update momentum for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['momentum'] = new_momentum

    return new_momentum
