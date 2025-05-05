import PIL
import random
import json
from collections import namedtuple
from math import inf
from pathlib import Path
from typing import Literal, Any, List

import torch
from pandas import read_csv, DataFrame, Series
from prettytable import PrettyTable
from torch import Tensor, eq, zeros
from torch import nn
from torch.nn.init import normal_, constant_
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms as T

Sample = namedtuple('Sample', ['X', 'Y'])


class CustomDataset(Dataset):
    def __init__(self, data_path: Path, portion: Literal["train", "valid", "test"], test_seed: int = 412):
        self.portion = portion
        self.data_path = data_path

        self.img2class: Series = (
            read_csv(data_path / f"annotated_{portion}.csv")
            .set_index("img_path")["class_idx"]
            .sample(frac=1, random_state=test_seed)
        )

        self.tfms = T.Compose([
            T.Resize((400, 400)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img2class)

    def __getitem__(self, idx: int) -> Sample:
        img_name = self.img2class.index[idx]
        img_path = self.data_path / img_name

        img_tensor = PIL.Image.open(img_path)
        img_tensor = self.tfms(img_tensor)

        return Sample(img_tensor, self.img2class.loc[img_name])


def get_available_accelerators()-> List[str]:
    accelerators = []

    # Check for CUDA (Linux/Windows with NVIDIA GPU)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            accelerators.append(name)

    # Check for MPS (macOS with Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerators.append("mps")

    accelerators.append("cpu")

    return accelerators

class TrackValues:
    def __init__(self):
        self.history = []

    def add(self, value: Any):
        self.history.append(value)

    def get_loss(self):
        return [*range(len(self.history))], torch.tensor(self.history)

    def last(self):
        return self.history[-1]


class EarlyStopping:
    """
    The early stopping it used to avoid the over-fitting.
    """

    def __init__(self, patience: int):
        self.patience: int = patience
        self.curr_pat: int = patience + 1
        self.current_vl: float = -inf
        self.earlyStop = False

    def update(self, vl_loss: float):
        if self.current_vl < vl_loss:
            self.curr_pat -= 1
        else:
            self.curr_pat = self.patience
        self.current_vl = vl_loss
        if self.curr_pat == 0:
            self.earlyStop = True

def compute_metrics(confusion: Tensor, all_metrics:bool=False):
    """
    Given a Confusion matrix, returns an F1-score, if all_metrics is false, then returns only a mean of F1-score
    """
    length = confusion.shape[0]
    iter_label = range(length)

    accuracy: Tensor = zeros(length)
    precision: Tensor = zeros(length)
    recall: Tensor = zeros(length)
    f1: Tensor = zeros(length)

    for i in iter_label:
        fn = torch.sum(confusion[i, :i]) + torch.sum(confusion[i, i + 1:])  # false negative
        fp = torch.sum(confusion[:i, i]) + torch.sum(confusion[i + 1:, i])  # false positive
        tn, tp = 0, confusion[i, i]  # true negative, true positive

        for x in iter_label:
            for y in iter_label:
                if (x != i) & (y != i):
                    tn += confusion[x, y]

        accuracy[i] = (tp + tn) / (tp + fn + fp + tn)
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    if all_metrics:
        return DataFrame({
            "Accuracy": accuracy.tolist(),
            "Precision": precision.tolist(),
            "Recall": recall.tolist(),
            "F1": f1.tolist()})
    else:
        return f1.nanmean()


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        normal_(m.weight.data, 0.0, 0.02)

    elif class_name.find("BatchNorm2d") != -1:
        normal_(m.weight.data, 1.0, 0.02)
        constant_(m.bias.data, 0.0)

def count_parameters(model):
    "https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model"
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

def update_cache(metrics:dict, output_file: Path):

    with output_file.open("r+") as file:
        file_data = json.load(file)
        file_data.append(metrics)
        file.seek(0)
        json.dump(file_data, file, indent=2)
