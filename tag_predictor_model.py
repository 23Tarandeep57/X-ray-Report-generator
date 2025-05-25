import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=14, freeze_until='denseblock4'):
    """
    Builds a DenseNet-121 model with the option to freeze layers up to a specified block.

    Args:
        num_classes (int): Number of output classes.
        freeze_until (str): Name of the layer up to which to freeze the model.
                            Options include 'denseblock1', 'denseblock2', 'denseblock3', 'denseblock4'.

    Returns:
        model (nn.Module): Modified DenseNet-121 model.
    """
    # Load the pretrained DenseNet-121 model
    model = models.densenet121(pretrained=True)

    # Freeze layers up to the specified block
    freeze = True
    for name, child in model.features.named_children():
        if name == freeze_until:
            freeze = False
        if freeze:
            for param in child.parameters():
                param.requires_grad = False

    # Replace the classifier to match the number of classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    return model
