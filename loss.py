import torch
import torch.nn as nn
import torch.nn.functional as F

def get_alpha_vector(labels):
    """
    Compute alpha per class based on class frequency.
    Rare classes get higher alpha.
    """
    pos_counts = labels.sum(axis=0)
    # neg_counts = labels.shape[0] - pos_counts
    alpha = 1.0 / (pos_counts + 1e-6)  # Avoid division by zero
    alpha = alpha / alpha.sum()  # Normalize to sum to 1 (optional)
    return torch.tensor(alpha, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: Tensor of shape (num_classes,) — per-class weights
        gamma: Focusing parameter for hard examples
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits (batch_size, num_classes)
        targets: binary labels (batch_size, num_classes)
        """
        probs = torch.sigmoid(inputs)
        targets = targets.type_as(inputs)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p if y == 1 else 1 - p
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * focal_term * BCE_loss
        else:
            loss = focal_term * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
