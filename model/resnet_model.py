from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


class CNNClassifier(nn.Module):
    """
    A customizable CNN classifier based on ResNet architectures.

    This class provides a convenient wrapper around torchvision's ResNet models
    (resnet18 and resnet34), allowing for easy initialization with pretrained
    weights, modification of the final classification layer, and freezing of
    early layers for transfer learning.

    Parameters:
    ------------
        num_classes (int): The number of output classes for the final
            classification layer.
        model_name (Literal["resnet18", "resnet34"]): The specific ResNet
            architecture to use.
        pretrained (bool, optional): If True, loads weights pretrained on
            ImageNet. Defaults to True.
        freeze_layers (int, optional): The number of initial layers to freeze
            The layers are considered the main sequential blocks of the ResNet
            model. Defaults to 0.
    """

    def __init__(self, num_classes: int, model_name: Literal["resnet18", "resnet34"] = "resnet18",
                 pretrained: bool = True, freeze_layers: int = 0):

        super().__init__()

        weights = 'DEFAULT' if pretrained else None

        model_builder = getattr(models, model_name)
        self.model = model_builder(weights=weights, progress=True)

        # Replace the fully connected layer for the new number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes, bias=True)

        if freeze_layers > 0:
            self.freeze_first_n_layers(freeze_layers)

    def freeze_first_n_layers(self, n_layers: int):
        """
        Freezes the first n layers of the model.

        The layers are the top-level children of the ResNet model, which
        typically include the initial conv, batch norm, relu, maxpool, and
        the subsequent sequential layer blocks.

        Args:
            n_layers (int): The number of initial layers to freeze.
        """
        # Get the top-level modules of the model
        layers = list(self.model.children())

        # Freeze the parameters for the first n layers
        for i, layer in enumerate(layers):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the classifier.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W), where
                N is the batch size, C is the number of channels, H is the
                height, and W is the width.

        Returns:
            torch.Tensor: The output tensor of shape (N, num_classes),
                representing the model's predictions.
        """
        return self.model(x)


# Example usage
if __name__ == '__main__':
    classifier = CNNClassifier(num_classes=2, model_name="resnet18", pretrained=True, freeze_layers=9)

    # Print the model architecture
    print(classifier)

    # Check which parameters are frozen
    for name, param in classifier.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    #dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    #output = classifier(dummy_input)

    #print(f"\nOutput shape: {output.shape}")