import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_utils import CustomDataset
from model.resnet_model import CNNClassifier
from training import train_classifier


def car_classifier(dataset_path:Path, metric_history:Path, model_cache:Path, new_hist:bool, batch_size:int=150,
                   model_name:str="resnet18", pretrained:bool=True, freeze_layers:int=0, num_epochs:int = 10,
                   device :str='cuda:0'):

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
                              batch_size=batch_size, shuffle=True, num_workers=4)

    cls = CNNClassifier(num_classes=196, model_name=model_name, pretrained=pretrained, freeze_layers=freeze_layers)

    if new_hist:
        with metric_history.open("w") as file:
            json.dump([], file)

    train_classifier(
        model=cls,
        dataset=(train_loader, valid_loader),


        optimizer_cls=torch.optim.AdamW,
        opt_params=dict(lr=0.0003, weight_decay=0.0001),

        scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
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
    Parses command-line arguments and calls the car_classifier function.
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
    print("Running car_classifier script...")
    main()