"""
================================================================================
Author: Andrea Iommi
Code Ownership:
    - All Python source code in this file is written solely by the author.
Documentation Notice:
    - All docstrings and inline documentation are written by ChatGPT,
      but thoroughly checked and approved by the author for accuracy.
================================================================================
"""

from typing import Literal

import torch.nn as nn
from torch import Tensor
from torchvision import models


class CNNClassifier(nn.Module):
    """
    A customizable CNN classifier based on ResNet architectures.

    This class provides a convenient wrapper around torchvision's ResNet models
    (resnet18 and resnet34), allowing for:
        - Initialization with pretrained weights (ImageNet)
        - Modification of the final classification layer
        - Freezing of early layers for transfer learning

    Parameters
    ----------
    num_classes : int
        The number of output classes for the final classification layer.

    model_name : Literal["resnet18", "resnet34"], optional, default="resnet18"
        The specific ResNet architecture to use.

    pretrained : bool, optional, default=True
        If True, loads weights pretrained on ImageNet.

    freeze_layers : int, optional, default=0
        The number of initial layers to freeze for transfer learning.
        Layers are considered as the top-level children of the ResNet model.
    """

    def __init__(self, num_classes: int,
                 model_name: Literal["resnet18", "resnet34"] = "resnet18",
                 pretrained: bool = True,
                 freeze_layers: int = 0):
        super().__init__()

        # Determine pretrained weights
        weights = 'DEFAULT' if pretrained else None

        # Dynamically load the requested ResNet model
        model_builder = getattr(models, model_name)
        self.model = model_builder(weights=weights, progress=True)

        # Replace the fully connected layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes, bias=True)

        # Freeze the first N layers if requested
        if freeze_layers > 0:
            self.freeze_first_n_layers(freeze_layers)

    def freeze_first_n_layers(self, n_layers: int):
        """
        Freezes the first `n_layers` of the model to prevent their weights
        from updating during training.

        Parameters
        ----------
        n_layers : int
            The number of initial layers to freeze.
        """
        layers = list(self.model.children())  # Get top-level modules

        # Freeze parameters of the first n layers
        for i, layer in enumerate(layers):
            if i < n_layers:
                for param_ in layer.parameters():
                    param_.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the CNN classifier.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W), where
            N = batch size, C = channels, H = height, W = width.

        Returns
        -------
        Tensor
            Output tensor of shape (N, num_classes), representing model predictions.
        """
        return self.model(x)


# Example usage
if __name__ == '__main__':
    classifier = CNNClassifier(
        num_classes=2,
        model_name="resnet18",
        pretrained=True,
        freeze_layers=9
    )

    # Print the model architecture
    print(classifier)

    # Check which parameters are frozen
    for name, param in classifier.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
