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

import json
from collections import namedtuple
from math import inf
from pathlib import Path
from typing import Literal, List

import PIL
import pandas as pd
import torch
from pandas import read_csv, Series
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

# Named tuple for returning dataset samples in a clean structure
Sample = namedtuple('Sample', ['X', 'Y'])


class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for loading image data and their corresponding class labels.
    It handles reading image paths from a CSV, applying transformations, and providing
    samples suitable for training or evaluation.
    """

    def __init__(self, data_path: Path, portion: Literal["train", "valid", "test"], test_seed: int = 412):
        """
        Initializes the CustomDataset.

        Parameters:
        -----------
        data_path (Path):
            The root directory where the dataset is located.
            This directory should contain 'annotated_{portion}.csv' files
            and the image files themselves.

        portion (Literal["train", "valid", "test"]):
            Specifies which subset of the data to load
            (training, validation, or test).

        test_seed (int, optional):
            A random seed for shuffling the dataset.
            Default is 412 to ensure reproducibility.
        """
        self.portion = portion
        self.data_path = data_path

        # Load annotations from CSV, shuffle them, and store mapping img_path -> class_idx
        self.img2class: Series = (
            read_csv(data_path / f"annotated_{portion}.csv")
            .set_index("img_path")["class_idx"]
            .sample(frac=1, random_state=test_seed)
        )

        # Define preprocessing transformations
        self.tfms = T.Compose([
            T.Resize((400, 400)),  # Resize to 400x400 pixels
            T.ToTensor(),  # Convert to torch.Tensor
            T.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        """
        Returns:
        --------
        int:
            The total number of samples in the dataset.
        """
        return len(self.img2class)

    def __getitem__(self, idx: int) -> Sample:
        """
        Retrieves a single image and its label at the given index.

        Parameters:
        -----------
        idx (int):
            Index of the sample to retrieve.

        Returns:
        --------
        Sample:
            A namedtuple containing:
              - X (torch.Tensor): Preprocessed image tensor.
              - Y (int): Class label of the image.
        """
        img_name = self.img2class.index[idx]
        img_path = self.data_path / img_name

        # Load and transform image
        img_tensor = PIL.Image.open(img_path)
        img_tensor = self.tfms(img_tensor)

        return Sample(img_tensor, self.img2class.loc[img_name])


def get_annotates_classes(data_path: Path) -> dict:
    """
    Loads class annotations from CSV and returns a mapping from
    class index to class name.

    Parameters:
    -----------
    data_path (Path):
        Path to the dataset root directory.

    Returns:
    --------
    dict:
        Dictionary mapping class_idx -> class_name.
    """
    return pd.read_csv(data_path / "annotated_classes.csv").set_index("class_idx")["class_name"].to_dict()


def get_available_accelerators() -> List[str]:
    """
    Detects available hardware accelerators for PyTorch.

    Returns:
    --------
    List[str]:
        A list of device strings:
        - "cuda:N" for NVIDIA GPUs (if available).
        - "mps" for Apple Silicon (if available).
        - "cpu" is always included as fallback.
    """
    accelerators = []

    # CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            accelerators.append(f"cuda:{i}")

    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerators.append("mps")

    # Always include CPU
    accelerators.append("cpu")

    return accelerators


class EarlyStopping:
    """
    Implements Early Stopping for training loops.

    Monitors validation loss and stops training if no improvement is observed
    for a specified number of epochs (`patience`).
    """

    def __init__(self, patience: int):
        """
        Parameters:
        -----------
        patience (int):
            Number of epochs to wait for improvement before stopping training.
        """
        self.patience: int = patience
        self.curr_pat: int = patience + 1  # Counter before triggering stop
        self.current_vl: float = -inf  # Track the best validation loss
        self.earlyStop = False  # Stop flag

    def update(self, vl_loss: float):
        """
        Update state with the current validation loss.

        Parameters:
        -----------
        vl_loss (float):
            Current validation loss.

        Notes:
        ------
        - If loss improves, patience counter is reset.
        - If not, patience counter decreases.
        - If patience reaches 0, training should stop.
        """
        if self.current_vl < vl_loss:
            self.curr_pat -= 1
        else:
            self.curr_pat = self.patience
        self.current_vl = vl_loss
        if self.curr_pat == 0:
            self.earlyStop = True


def update_cache(metrics: dict, output_file: Path):
    """
    Append new metrics to a JSON log file.

    Parameters:
    -----------
    metrics (dict):
        Dictionary of training/validation metrics.

    output_file (Path):
        Path to JSON file for caching results.

    Notes:
    ------
    - If file does not exist, an empty list is created.
    - Appends metrics sequentially for experiment tracking.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not output_file.exists() or output_file.stat().st_size == 0:
        with output_file.open("w") as file:
            json.dump([], file)

    with output_file.open("r+") as file:
        file_data = json.load(file)
        file_data.append(metrics)
        file.seek(0)
        json.dump(file_data, file, indent=2)
