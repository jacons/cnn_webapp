import datetime
import time
from pathlib import Path
from typing import Tuple, Type, Dict

import sklearn.metrics
import torch
from pandas import DataFrame, Series
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml_utils import EarlyStopping, update_cache


def eval_class_errors(dataloader: DataLoader, model: nn.Module,  device: str) -> DataFrame:

    y_pred, y_trues = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(model(x), dim=1)
            y_pred.append(pred)
            y_trues.append(y)

    y_pred = torch.cat(y_pred).cpu().numpy()
    y_trues = torch.cat(y_trues).cpu().numpy()
    df = DataFrame({'y_pred': y_pred, 'y_true': y_trues})

    def compute_group_stats(group: DataFrame):
        correct = (group.y_pred == group.y_true).sum()
        total = len(group)
        return Series({'right': correct, 'total': total, 'frac': correct / total})

    return df.groupby("y_true").apply(compute_group_stats)


def eval_dataset(
        dataloader: DataLoader,
        model: nn.Module,
        criterion: CrossEntropyLoss,
        metrics: dict,
        device: str,
        evaluation: str,
        p_bar: tqdm = None,
        epoch: int = None,
        num_epochs: int = None,
        total_batches: int = None,
        current_state: Dict = None):

    model.eval()
    losses, preds, trues = [], [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            predictions = torch.argmax(logits, dim=1)

            losses.append(loss)
            preds.append(predictions)
            trues.append(y)

            if current_state:
                current_state["batch"] = f"Batch {idx}/{total_batches}"
            if p_bar:
                p_bar.set_description(f"[{evaluation}] epoch {epoch}/{num_epochs} | batch {idx}/{total_batches}")

    preds = torch.cat(preds).cpu().numpy()
    trues = torch.cat(trues).cpu().numpy()
    losses = torch.stack(losses)

    prefix = evaluation[:5]
    metrics[f"{prefix}_loss"] = losses.mean().item()
    metrics[f"{prefix}_loss_std"] = losses.std().item()
    metrics[f"{prefix}_f1"] = sklearn.metrics.f1_score(trues, preds, average='weighted')
    metrics[f"{prefix}_acc"] = sklearn.metrics.accuracy_score(trues, preds)


def train_classifier(
        model: nn.Module,
        dataset: Tuple[DataLoader, DataLoader],
        optimizer: Type[optim.Optimizer],
        opt_params: dict,
        scheduler: Type[optim.lr_scheduler],
        sc_params: dict,
        num_epochs: int,
        metric_history: Path,
        model_cache: str,
        patience: int,
        device: str,
        current_state: Dict):

    train_dl, valid_dl = dataset
    es = EarlyStopping(patience if patience > 0 else num_epochs)

    model.to(device)
    criterion = CrossEntropyLoss().to(device)
    optimizer = optimizer(model.parameters(), **opt_params)
    scheduler = scheduler(optimizer, **sc_params) if scheduler else None

    train_batches = len(train_dl)
    valid_batches = len(valid_dl)

    start_time = time.time()
    current_state["time_start"] = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

    p_bar = tqdm(range(num_epochs), desc="[Training]")

    for epoch in p_bar:
        if es.earlyStop:
            break

        model.train()
        current_state["epoch"] = f"[Training] epoch {epoch}/{num_epochs}"

        for idx, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            current_state["batch"] = f"Batch {idx}/{train_batches}"
            p_bar.set_description(f"[Training] epoch {epoch}/{num_epochs} | batch {idx}/{train_batches}")

        # Validation
        current_state["epoch"] = f"[Validation] epoch {epoch}/{num_epochs}"
        metrics = dict()

        eval_dataset(train_dl, model, criterion, metrics, device, "training", p_bar, epoch, num_epochs, train_batches, current_state)
        eval_dataset(valid_dl, model, criterion, metrics, device, "validation", p_bar, epoch, num_epochs, valid_batches, current_state)

        # Save results
        update_cache(metrics, metric_history)
        torch.save(model.cpu().state_dict(), Path(model_cache))

        es.update(metrics["valid_f1"] * -1)  # Lower is better for EarlyStopping

        p_bar.set_postfix(
            tr_loss=metrics["train_loss"],
            vl_loss=metrics["valid_loss"],
            tr_f1=metrics["train_f1"],
            vl_f1=metrics["valid_f1"]
        )

        current_state.update({
            "status": int(100 * (p_bar.n / p_bar.total)),
            "tr_loss_mean": metrics["train_loss"],
            "tr_loss_std": metrics["train_loss_std"],
            "vl_loss_mean": metrics["valid_loss"],
            "vl_loss_std": metrics["valid_loss_std"],
            "vl_f1": metrics["train_f1"],
            "vl_acc": metrics["valid_f1"],
            "time_elapsed": str(datetime.timedelta(seconds=int(time.time() - start_time)))
        })