import json
from collections import namedtuple
from math import inf
from pathlib import Path
from typing import Literal, List

import PIL
import pandas as pd
import torch
from pandas import read_csv, Series
from prettytable import PrettyTable
from torch.nn.init import normal_, constant_
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

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
            data_path (Path): The root directory where the dataset is located.
                              This directory should contain 'annotated_{portion}.csv' files
                              and the image files themselves.
            portion (Literal["train", "valid", "test"]): Specifies which subset of the data
                                                        to load (training, validation, or test).
            test_seed (int, optional): A seed for random sampling of the dataset.
        """
        self.portion = portion
        self.data_path = data_path

        # Read the annotation CSV, set 'img_path' as index, get 'class_idx',
        # and shuffle the DataFrame for randomness.
        self.img2class: Series = (
            read_csv(data_path / f"annotated_{portion}.csv")
            .set_index("img_path")["class_idx"]
            .sample(frac=1, random_state=test_seed)
        )

        # Define image transformations to be applied to each image.
        self.tfms = T.Compose([
            T.Resize((400, 400)),  # Resize images to 400x400 pixels
            T.ToTensor(),          # Convert PIL Image to PyTorch Tensor
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize image pixel values
        ])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        ---------
            int: The number of images in the dataset.
        """
        return len(self.img2class)

    def __getitem__(self, idx: int) -> Sample:
        """
        Retrieves a single sample (image and its label) from the dataset at the given index.

        Parameters:
        -----------
            idx (int): The index of the sample to retrieve.

        Returns:
        ----------
            Sample: A namedtuple containing:
                    - X (torch.Tensor): The transformed image tensor.
                    - Y (int): The class label for the image.
        """
        img_name = self.img2class.index[idx]
        img_path = self.data_path / img_name

        # Open the image using PIL, apply transformations, and get the class label.
        img_tensor = PIL.Image.open(img_path)
        img_tensor = self.tfms(img_tensor)

        return Sample(img_tensor, self.img2class.loc[img_name])

def get_annotates_classes(data_path: Path) -> dict:
    """
    Reads a CSV file containing class annotations and returns a dictionary mapping
    :param data_path:
    :return:
    """
    return pd.read_csv(data_path / "annotated_classes.csv").set_index("class_idx")["class_name"].to_dict()

def get_available_accelerators() -> List[str]:
    """
    Detects and returns a list of available hardware accelerators for PyTorch.
    This includes CUDA GPUs (NVIDIA), MPS (Apple Silicon), and CPU.

    Returns:
    ----------
        List[str]: A list of strings, where each string represents an available
                   accelerator (e.g., "cuda:0", "mps", "cpu").
    """
    accelerators = []

    # Check for CUDA (Linux/Windows with NVIDIA GPU)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            accelerators.append(f"cuda:{i}")

    # Check for MPS (macOS with Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerators.append("mps")

    # CPU is always available as a fallback
    accelerators.append("cpu")

    return accelerators


class EarlyStopping:
    """
    The EarlyStopping class is used to monitor a validation metric (e.g., validation loss)
    during model training and stop the training process if the metric does not improve
    for a specified number of epochs (patience). This helps to prevent overfitting.
    """

    def __init__(self, patience: int):
        """
        Initializes the EarlyStopping instance.

        Parameters:
        -----------
            patience (int): The number of epochs to wait for improvement before stopping.
                            If the validation loss does not decrease for `patience` consecutive
                            epochs, training will be stopped.
        """
        self.patience: int = patience
        self.curr_pat: int = patience + 1 # Initialize to allow at least one update before checking
        self.current_vl: float = -inf      # Stores the best validation loss seen so far
        self.earlyStop = False             # Flag to indicate if early stopping condition is met

    def update(self, vl_loss: float):
        """
        Updates the early stopping state with the current validation loss.

        Args:
            vl_loss (float): The current validation loss from the training epoch.
        """
        if self.current_vl < vl_loss:
            # If the current loss is worse than the best seen, decrement patience counter
            self.curr_pat -= 1
        else:
            # If the current loss is better or equal, reset patience counter
            self.curr_pat = self.patience
        self.current_vl = vl_loss # Update the best validation loss
        if self.curr_pat == 0:
            # If patience runs out, set earlyStop flag to True
            self.earlyStop = True


def weights_init_normal(m):
    """
    Initializes the weights of convolutional and batch normalization layers
    in a PyTorch model with a normal distribution.

    - Convolutional layers (Conv): Weights are initialized from a normal distribution
      with mean 0.0 and standard deviation 0.02.
    - BatchNorm2d layers: Weights are initialized from a normal distribution
      with mean 1.0 and standard deviation 0.02, and biases are set to 0.0.

    This function is typically applied using `model.apply(weights_init_normal)`.

    Args:
        m (torch.nn.Module): A module from a PyTorch model.
    """
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        normal_(m.weight.data, 0.0, 0.02)

    elif class_name.find("BatchNorm2d") != -1:
        normal_(m.weight.data, 1.0, 0.02)
        constant_(m.bias.data, 0.0)

def count_parameters(model):
    """
    Counts and prints the number of trainable parameters in a PyTorch model.
    It also returns the total count of trainable parameters.

    Reference: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    Args:
        model (torch.nn.Module): The PyTorch model for which to count parameters.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def update_cache(metrics: dict, output_file: Path):
    """
    Appends a dictionary of metrics to a JSON file. If the file is empty or does not exist,
    it will be initialized as an empty list before appending.

    Args:
        metrics (dict): A dictionary containing the metrics to be added to the cache.
        output_file (Path): The path to the JSON file where the metrics will be stored.
    """
    # Ensure the parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # If a file doesn't exist or is empty, initialize with an empty list
    if not output_file.exists() or output_file.stat().st_size == 0:
        with output_file.open("w") as file:
            json.dump([], file)

    with output_file.open("r+") as file:
        file_data = json.load(file)
        file_data.append(metrics)
        file.seek(0)  # Rewind to the beginning of the file
        json.dump(file_data, file, indent=2) # Write updated data back to file with indentation
