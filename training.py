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

import datetime
import time
from pathlib import Path
from typing import Tuple, Type, Dict

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml_utils import EarlyStopping, update_cache


def evaluate_model(dataloader: DataLoader, model: nn.Module, metrics: dict,
                   device: str, phase: str, p_bar: tqdm, total_batches: int, state: Dict) -> None:
    """
    Evaluates the model's performance on a given dataset.

    The function sets the model to evaluation mode, iterates through the provided
    dataloader, performs forward passes, computes predictions, and calculates
    evaluation metrics such as mean loss, standard deviation of loss, F1-score,
    and accuracy. Results are stored in the provided `metrics` dictionary.

    Parameters
    ----------
    dataloader : DataLoader
        A DataLoader providing batches of data for evaluation.

    model : nn.Module
        The PyTorch model to be evaluated.

    metrics : dict
        A dictionary to store the calculated metrics. Keys will be prefixed with the
        first 5 characters of `phase` (e.g., "train_loss", "valid_acc").

    device : str
        The device ("cpu", "cuda:0", "mps") on which to perform evaluation.

    phase : str
        The current evaluation phase (e.g., "train", "valid", "test").
        Used for metric key prefixes.

    p_bar : tqdm
        A tqdm progress bar instance used to update progress descriptions.

    total_batches : int
        Total number of batches in the dataloader (used for progress updates).

    state : Dict
        A dictionary containing training state information, including "epoch"
        and "batch". Updated to reflect evaluation progress.

    Returns
    -------
    None
        Results are stored directly in the `metrics` dictionary.
    """
    model.eval()  # Set the model to evaluation mode
    losses, preds, trues = [], [], []
    criterion = CrossEntropyLoss().to(device)

    with torch.no_grad():  # Disable gradient tracking during evaluation
        for idx, (x, y) in enumerate(dataloader, 1):
            if device != "cpu":
                x, y = x.to(device), y.to(device)  # Move batch to the specified device
            logits = model(x)  # Forward pass
            loss = criterion(logits, y)  # Compute loss
            predictions = torch.argmax(logits, dim=1)  # Get predicted class indices

            losses.append(loss)
            preds.append(predictions)
            trues.append(y)

            # Update progress bar
            if state:
                state["batch"] = f"Batch {idx}/{total_batches}"
            if p_bar:
                p_bar.set_description(f"{state['epoch']} | {state['batch']}")

    # Aggregate predictions, labels, and losses
    preds = torch.cat(preds).cpu().numpy()
    trues = torch.cat(trues).cpu().numpy()
    losses_tensor = torch.stack(losses)

    # Compute metrics and update dictionary
    prefix = phase[:5]  # Metric key prefix based on phase
    metrics[f"{prefix}_loss"] = losses_tensor.mean().item()
    metrics[f"{prefix}_loss_std"] = losses_tensor.std().item()
    metrics[f"{prefix}_f1"] = f1_score(trues, preds, average='weighted')
    metrics[f"{prefix}_acc"] = accuracy_score(trues, preds)


def train_classifier(model: nn.Module, dataset: Tuple[DataLoader, DataLoader], optimizer_cls: Type[optim.Optimizer],
                     opt_params: dict, scheduler_cls: Type[optim.lr_scheduler.LRScheduler], scheduler_params: dict,
                     num_epochs: int, metric_history: Path, model_cache: Path, patience: int, device: str,
                     state: Dict) -> None:
    """
    Trains a PyTorch classifier model with support for:
    - Early stopping
    - Learning rate scheduling
    - Metric tracking (loss, F1-score, accuracy)
    - Progress bar visualization

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.

    dataset : Tuple[DataLoader, DataLoader]
        A tuple containing (training DataLoader, validation DataLoader).

    optimizer_cls : Type[optim.Optimizer]
        Optimizer class (e.g., torch.optim.Adam).

    opt_params : dict
        Parameters to initialize the optimizer.

    scheduler_cls : Type[optim.lr_scheduler._LRScheduler] or None
        Learning rate scheduler class. Can be None if no scheduler is desired.

    scheduler_params : dict
        Parameters to initialize the scheduler.

    num_epochs : int
        Number of epochs to train for.

    metric_history : Path
        Path to a JSON file where metrics will be logged per epoch.

    model_cache : Path
        Path where the best model state dictionary will be saved.

    patience : int
        Number of epochs to wait for improvement in validation F1-score
        before stopping early. If ≤ 0, training runs for all epochs.

    device : str
        Device ("cpu", "cuda:0", "mps") for training and evaluation.

    state : Dict
        A mutable dictionary to share training status externally.
        Includes epoch, batch, loss, F1-score, and elapsed time.

    Returns
    -------
    None
        Training progress and results are tracked through files, logs,
        and the provided `state` dictionary.
    """
    train_dl, valid_dl = dataset

    # Initialize EarlyStopping. If patience ≤ 0, it effectively runs for all epochs.
    es = EarlyStopping(patience if patience > 0 else num_epochs)

    model.to(device)  # Move the model to the device
    criterion = CrossEntropyLoss().to(device)  # Define loss function
    optimizer = optimizer_cls(model.parameters(), **opt_params)  # Initialize optimizer
    scheduler = scheduler_cls(optimizer, **scheduler_params) if scheduler_cls else None  # LR scheduler

    # Track training start time
    start_time = time.time()
    state["time_start"] = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

    p_bar = tqdm(range(num_epochs), desc="[Training]")  # Epoch progress bar
    train_batches, valid_batches = len(train_dl), len(valid_dl)

    for epoch in p_bar:
        if es.earlyStop:  # Early stopping check
            print(f"Early stopping triggered at epoch {epoch}. No improvement in validation F1-score.")
            break

        # Training loop
        model.train()
        state["epoch"] = f"[Training] epoch {epoch}/{num_epochs}"

        for idx, (x, y) in enumerate(train_dl, 1):
            if device != "cpu":
                x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            # Update progress bar with batch info
            state["batch"] = f"Batch {idx}/{train_batches}"
            p_bar.set_description(f"{state['epoch']} | {state['batch']}")
            progress_bar = int(100 * ((idx + train_batches * epoch) / (num_epochs * train_batches)))
            state.update({"status": progress_bar})

        if scheduler:
            scheduler.step()  # Step the scheduler (if defined)

        # Evaluation loop
        state["epoch"] = f"[Validation] epoch {epoch}/{num_epochs}"
        metrics = {}

        evaluate_model(train_dl, model, metrics, device, "training", p_bar, train_batches, state)
        evaluate_model(valid_dl, model, metrics, device, "validation", p_bar, valid_batches, state)

        # Save metrics and model state
        update_cache(metrics, metric_history)
        torch.save(model.cpu().state_dict(), model_cache)
        model.to(device)  # Reload model back to device if saved on CPU

        # Update early stopping with negative F1-score (minimization target)
        es.update(metrics["valid_f1"] * -1)

        # Update progress bar postfix
        p_bar.set_postfix(
            tr_loss=metrics["train_loss"],
            vl_loss=metrics["valid_loss"],
            tr_f1=metrics["train_f1"],
            vl_f1=metrics["valid_f1"]
        )

        # Update shared state dictionary
        state.update({
            "status": int(100 * (p_bar.n / p_bar.total)),  # Training % progress
            "tr_loss_mean": round(metrics["train_loss"], 4),
            "tr_loss_std": round(metrics["train_loss_std"], 4),
            "vl_loss_mean": round(metrics["valid_loss"], 4),
            "vl_loss_std": round(metrics["valid_loss_std"], 4),
            "vl_f1": round(metrics["valid_f1"], 4),
            "vl_acc": round(metrics["valid_acc"], 4),
            "time_elapsed": str(datetime.timedelta(seconds=int(time.time() - start_time)))
        })