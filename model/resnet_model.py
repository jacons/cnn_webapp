import torch.nn as nn
from torchvision.models import resnet18


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):

        super(CNNClassifier, self).__init__()

        weights = 'DEFAULT' if pretrained else None

        self.resnet = resnet18(weights=weights, progress=True)
        self.resnet.fc = nn.Linear(512, num_classes, bias=True)

    def freeze_first_n_layers(self, n_layers):

        layers = list(self.resnet.children())  # Get the model's layers as a list

        # Flatten nested layers
        all_layers = []
        for layer in layers:
            if isinstance(layer, nn.Sequential):
                all_layers.extend(layer.children())
            else:
                all_layers.append(layer)

        # Freeze the first n layers
        for layer in all_layers[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)