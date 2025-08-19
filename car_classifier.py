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

import argparse
import json
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ml_utils import CustomDataset
from model.resnet_model import CNNClassifier
from training import train_classifier


def car_classifier(dataset_path:Path, metric_history:Path, model_cache:Path, new_hist:bool, batch_size:int=150,
                   model_name:str="resnet18", pretrained:bool=True, freeze_layers:int=0, num_epochs:int = 10,
                   device :str='cuda:0'):
    """
    Trains a car classification model using a ResNet-based CNN architecture.

    Parameters:
    -----------
    dataset_path (Path):
        Path to the root dataset directory containing the train/valid/test splits.
        Each split must be defined via CSV annotations and associated images.

    metric_history (Path):
        File path where the training and validation metrics will be saved in JSON format.
        This allows resuming and tracking model performance across runs.

    model_cache (Path):
        File path where the trained model checkpoint (.pth file) will be saved.

    new_hist (bool):
        If True, a new empty metric history file is created, overwriting any existing file.

    batch_size (int, default=150):
        Number of samples per batch during training and validation.

    model_name (str, default="resnet18"):
        The ResNet architecture variant to use (e.g., "resnet18", "resnet34", "resnet50").

    pretrained (bool, default=True):
        If True, loads a pretrained ResNet model from torchvision as the base.

    freeze_layers (int, default=0):
        Number of initial layers to freeze (useful when fine-tuning pretrained models).

    num_epochs (int, default=10):
        Number of full training epochs.

    device (str, default="cuda:0"):
        Target device for training (e.g., "cpu", "cuda", "mps").
    """
    print("--- Starting car_classifier with the following parameters ---")
    print(f"Dataset Path: {dataset_path}")
    print(f"Metric History Path: {metric_history}")
    print(f"Model Cache Path: {model_cache}")
    print(f"New History: {new_hist}")
    print(f"Batch Size: {batch_size}")
    print(f"Model Name: {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"Freeze Layers: {freeze_layers}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Device: {device}")
    print("----------------------------------------------------------")
    dataset_path.mkdir(parents=True, exist_ok=True)
    metric_history.parent.mkdir(parents=True, exist_ok=True)
    model_cache.parent.mkdir(parents=True, exist_ok=True)

    train_dt = CustomDataset(data_path=dataset_path, portion="train")
    train_loader = DataLoader(train_dt,
                              batch_size=batch_size, shuffle=True, num_workers=4)

    valid_dt = CustomDataset(data_path=dataset_path, portion="valid")
    valid_loader = DataLoader(valid_dt,
                              batch_size=batch_size, shuffle=False, num_workers=4)

    cls = CNNClassifier(num_classes=196, model_name=model_name, pretrained=pretrained, freeze_layers=freeze_layers)

    if new_hist:
        with metric_history.open("w") as file:
            json.dump([], file)

    train_classifier(
        model=cls,
        dataset=(train_loader, valid_loader),


        optimizer_cls=AdamW,
        opt_params=dict(lr=0.0003, weight_decay=0.0001),

        scheduler_cls=CosineAnnealingLR,
        scheduler_params=dict(T_max=num_epochs, eta_min=1e-6),

        num_epochs=num_epochs,
        metric_history=metric_history,
        model_cache=model_cache,
        patience=5,
        device=device,
        state=dict()
    )


def main():
    """
        Entry point for the car classifier training script.

        Documentation Note:
        -------------------
        This documentation is written by ChatGPT but reviewed and verified by the author.
        The code itself is authored exclusively by the project’s author.

        Functionality:
        --------------
        - Parses command-line arguments for dataset paths, training options, and model settings.
        - Calls the `car_classifier` function with parsed arguments.
        - Enables training from terminal with flexible configuration.

        Example Usage:
        --------------
        python car_classifier.py    --dataset_path datasets/car_dataset
                                    --metric_history saved_models/resnet18/hist.json
                                    --model_cache saved_models/resnet18/model.pth
                                    --num_epochs 20
                                    --batch_size 128
                                    --device mps
        ```

    """
    parser = argparse.ArgumentParser(description="Train a car classifier model.")

    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="datasets/car_dataset",
        help="Path to the dataset directory (e.g., 'datasets/car_dataset')."
    )
    parser.add_argument(
        "--metric_history",
        type=Path,
        default="saved_models/resnet18/hist.json",
        help="Path to save metric history JSON file (e.g., 'saved_models/resnet18/hist.json')."
    )
    parser.add_argument(
        "--model_cache",
        type=Path,
        default="saved_models/resnet18/model.pth",
        help="Path to cache the trained model's state dictionary (e.g., 'saved_models/resnet18/model.pth')."
    )
    parser.add_argument(
        "--new_hist",
        action="store_true",
        help="If set, a new metric history file will be created, overwriting existing one."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=150,
        help="Batch size for training (default: 150)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="Name of the model architecture to use (default: 'resnet18')."
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="If set, use a pretrained model from torchvision."
    )
    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=0,
        help="Number of initial layers to freeze if using a pretrained model (default: 0)."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu", # Default to 'cpu', user can specify 'cuda' or 'mps'
        help="Device to run the training on (e.g., 'cpu', 'cuda', 'mps')."
    )

    args = parser.parse_args()

    # Call the car_classifier function with parsed arguments
    car_classifier(
        dataset_path=args.dataset_path,
        metric_history=args.metric_history,
        model_cache=args.model_cache,
        new_hist=args.new_hist,
        batch_size=args.batch_size,
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_layers=args.freeze_layers,
        num_epochs=args.num_epochs,
        device=args.device
    )

if __name__ == "__main__":
    main()