import datetime
import time
from pathlib import Path
from typing import Tuple, Type, Dict

import sklearn
import torch
from torch import nn, optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml_utils import EarlyStopping, update_cache


# def compute_class_errors(dataloader: DataLoader, model: nn.Module, device: str) -> DataFrame:
#     """Evaluate per-class prediction errors."""
#     y_pred, y_true = [], []
#
#     model.eval()
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             pred = torch.argmax(model(x), dim=1)
#             y_pred.append(pred)
#             y_true.append(y)
#
#     y_pred = torch.cat(y_pred).cpu().numpy()
#     y_true = torch.cat(y_true).cpu().numpy()
#     df = DataFrame({'y_pred': y_pred, 'y_true': y_true})
#
#     def group_stats(group: DataFrame) -> Series:
#         correct = (group.y_pred == group.y_true).sum()
#         total = len(group)
#         return Series({'right': correct, 'total': total, 'frac': correct / total})
#
#     return df.groupby("y_true").apply(group_stats)


def evaluate_model(dataloader: DataLoader, model: nn.Module, metrics: dict,
                   device: str, phase: str, p_bar: tqdm, total_batches: int, state: Dict) -> None:
    """
    Evaluates the model's performance on a given dataset.

    The function sets the model to evaluation mode, iterates through the provided
    dataloader, performs a forward pass, calculates the loss, and computes predictions.
    It then aggregates losses, predictions, and true labels to calculate and store
    evaluation metrics such as mean loss, standard deviation of loss, F1-score,
    and accuracy. The progress bar description is updated during iteration.

    Parameters:
    ----------
        dataloader (DataLoader): The DataLoader providing batches of data for evaluation.
        model (nn.Module): The PyTorch model to be evaluated.
        metrics (dict): A dictionary to store the calculated metrics.
                        Keys will be prefixed with the first 5 characters of `phase`
                        (e.g., "valid_loss", "test_acc").
        device (str): The device ("cpu", "cuda:0", "mps") on which to perform the evaluation.
        phase (str): The current evaluation phase (e.g., "train", "valid", "test").
                     Used for metric key prefixes.
        p_bar (tqdm): A tqdm progress bar instance to update its description.
        total_batches (int): The total number of batches in the dataloader.
        state (Dict): A dictionary containing training state information,
                      including "epoch" and "batch" which are updated for the progress bar.
    """
    model.eval()  # Set the model to evaluation mode
    losses, preds, trues = [], [], []
    criterion = CrossEntropyLoss().to(device)

    with torch.no_grad():  # Disable gradient calculations for inference
        for idx, (x, y) in enumerate(dataloader, 1):
            if device != "cpu":
                x, y = x.to(device), y.to(device)  # Move data to the specified device
            logits = model(x)  # Forward pass
            loss = criterion(logits, y)  # Calculate loss
            predictions = torch.argmax(logits, dim=1)  # Get class predictions

            losses.append(loss)
            preds.append(predictions)
            trues.append(y)

            # Update progress bar description
            if state:
                state["batch"] = f"Batch {idx}/{total_batches}"
            if p_bar:
                p_bar.set_description(f"{state['epoch']} | {state['batch']}")

    # Concatenate all predictions, true labels, and losses
    preds = torch.cat(preds).cpu().numpy()
    trues = torch.cat(trues).cpu().numpy()
    losses_tensor = torch.stack(losses)

    # Calculate and store metrics
    prefix = phase[:5] # Use the first 5 characters of phase for metric prefix
    metrics[f"{prefix}_loss"] = losses_tensor.mean().item()
    metrics[f"{prefix}_loss_std"] = losses_tensor.std().item()
    # Note: sklearn.metrics needs to be imported for f1_score and accuracy_score
    metrics[f"{prefix}_f1"] = sklearn.metrics.f1_score(trues, preds, average='weighted')
    metrics[f"{prefix}_acc"] = sklearn.metrics.accuracy_score(trues, preds)



def train_classifier(model: nn.Module, dataset: Tuple[DataLoader, DataLoader], optimizer_cls: Type[optim.Optimizer],
                     opt_params: dict, scheduler_cls: Type[optim.lr_scheduler._LRScheduler], scheduler_params: dict,
                     num_epochs: int, metric_history: Path, model_cache: Path, patience: int, device: str,
                     state: Dict) -> None:
    """
        Trains a PyTorch classifier model with support for early stopping,
        learning rate scheduling, and metric tracking.

        Parameters:
        ----------
            model (nn.Module): The PyTorch model to be trained.
            dataset (Tuple[DataLoader, DataLoader]): A tuple containing two DataLoaders:
                                                     (training DataLoader, validation DataLoader).
            optimizer_cls (Type[optim.Optimizer]): The class of the optimizer to use (e.g., torch.optim.Adam).
            opt_params (dict): A dictionary of parameters to pass to the optimizer's constructor.
            scheduler_cls (Type[optim.lr_scheduler._LRScheduler]): The class of the learning rate scheduler
                                                                    to use (e.g., torch.optim.lr_scheduler.StepLR).
                                                                    it can be None if no scheduler is desired.
            scheduler_params (dict): A dictionary of parameters to pass to the scheduler's constructor.
            num_epochs (int): The total number of epochs to train for.
            metric_history (Path): The file path (Path object) to a JSON file where
                                   epoch-wise training and validation metrics will be saved.
            model_cache (str): The file path (string) where the best model's state dictionary
                               will be saved.
            patience (int): The number of epochs to wait for improvement in validation F1-score
                            before triggering early stopping. If 0 or less, early stopping is effectively
                            disabled, and training runs for `num_epochs`.
            device (str): The device ("cpu", "cuda:0", "mps") on which to run the training.
            state (Dict): A mutable dictionary used to communicate the current training status
                          (e.g., epoch, batch, loss, F1-score, time elapsed) to an external caller.
        """
    train_dl, valid_dl = dataset

    # Initialize EarlyStopping. If patience is 0 or less, it effectively runs for all epochs.
    es = EarlyStopping(patience if patience > 0 else num_epochs)

    model.to(device)  # Move the model to the specified device
    criterion = CrossEntropyLoss().to(device)  # Initialize CrossEntropyLoss and move to a device
    optimizer = optimizer_cls(model.parameters(), **opt_params)  # Initialize optimizer
    # Initialize scheduler if a class is provided
    scheduler = scheduler_cls(optimizer, **scheduler_params) if scheduler_cls else None


    start_time = time.time()
    # Record the start time in a human-readable format
    state["time_start"] = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

    p_bar = tqdm(range(num_epochs), desc="[Training]") # Initialize progress bar
    train_batches,valid_batches = len(train_dl), len(valid_dl)
    for epoch in p_bar:
        if es.earlyStop: # Check if early stopping condition is met
            print(f"Early stopping triggered at epoch {epoch}. No improvement in validation F1-score.")
            break

        model.train() # Set the model to training mode
        state["epoch"] = f"[Training] epoch {epoch}/{num_epochs}"  # Update state for progress bar

        for idx, (x, y) in enumerate(train_dl, 1):
            if device != "cpu":
                x, y = x.to(device), y.to(device)  # Move data to a device
            optimizer.zero_grad()  # Zero the gradients
            loss = criterion(model(x), y)  # Forward pass and loss calculation
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            # Update progress bar description with current batch info
            state["batch"] = f"Batch {idx}/{train_batches}"
            p_bar.set_description(f"{state['epoch']} | {state['batch']}")

        if scheduler:
            scheduler.step()  # Step the scheduler (if defined) after each epoch

            # Evaluation phase
        state["epoch"] = f"[Validation] epoch {epoch}/{num_epochs}"
        metrics = {}  # Dictionary to store metrics for the current epoch

        # Evaluate on training and validation sets
        evaluate_model(train_dl, model, metrics, device, "training", p_bar,  train_batches, state)
        evaluate_model(valid_dl, model, metrics, device, "validation", p_bar, valid_batches, state)

        update_cache(metrics, metric_history)  # Save metrics to JSON file
        # Save the model's state dictionary (weights) to cache
        torch.save(model.cpu().state_dict(), model_cache)
        model.to(device)  # Move the model back to a device if it was moved to CPU for saving

        # Update early stopping with validation F1-score (negated because EarlyStopping expects decreasing loss)
        es.update(metrics["valid_f1"] * -1)

        # Update progress bar postfix with current epoch's metrics
        p_bar.set_postfix(
            tr_loss=metrics["train_loss"],
            vl_loss=metrics["valid_loss"],
            tr_f1=metrics["train_f1"],
            vl_f1=metrics["valid_f1"]
        )

        # Update the shared state dictionary with detailed metrics and time elapsed
        state.update({
            "status": int(100 * (p_bar.n / p_bar.total)),  # Training progress percentage
            "tr_loss_mean": metrics["train_loss"],
            "tr_loss_std": metrics["train_loss_std"],
            "vl_loss_mean": metrics["valid_loss"],
            "vl_loss_std": metrics["valid_loss_std"],
            "vl_f1": metrics["valid_f1"],  # Corrected from tr_f1 to vl_f1 for consistency
            "vl_acc": metrics["valid_acc"],  # Corrected from valid_f1 to valid_acc
            "time_elapsed": str(datetime.timedelta(seconds=int(time.time() - start_time)))
        })