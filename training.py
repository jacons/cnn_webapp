import datetime
import time
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Type, Dict

import sklearn
import sklearn.metrics
import torch
from pandas import DataFrame, Series
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml_utils import EarlyStopping, update_cache


def eval_class_errors(dataloader: DataLoader, model: nn.Module, device: str) -> DataFrame:

    y_pred_tmp, y_true_tmp, = [], []

    with torch.no_grad():
        for _, (x, y_true) in enumerate(dataloader, 1):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            class_pred = torch.argmax(y_pred, dim=1)

            y_pred_tmp.append(class_pred)
            y_true_tmp.append(y_true)

    y_pred_tmp = torch.concatenate(y_pred_tmp).cpu().numpy()
    y_true_tmp = torch.concatenate(y_true_tmp).cpu().numpy()

    errors = DataFrame(dict(y_pred=y_pred_tmp, y_true=y_true_tmp))

    return errors.groupby("y_true").apply(
        lambda class_: Series([
            ok := sum((class_.y_true == class_.y_pred).astype(int)),
            total := len(class_),
            ok / total
        ], index=["right", "total", "frac"])
    )


def eval_dataset(dataloader: DataLoader,
                 model: nn.Module,
                 criterion: nn.CrossEntropyLoss,
                 metrics: dict,
                 device: str,
                 evaluation: str,
                 p_bar: tqdm = None,
                 epoch: int = None,
                 num_epochs: int = None,
                 train_batches: int = None,
                 current_state: Dict = None):

    y_pred_tmp, y_true_tmp, loss_tmp = [], [], []

    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader, 1):

            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            loss_ = criterion(y_pred, y_true)
            class_pred = torch.argmax(y_pred, dim=1)

            loss_tmp.append(loss_)
            y_pred_tmp.append(class_pred)
            y_true_tmp.append(y_true)

            current_state["batch"] = f"Batch {idx}/{train_batches}"
            if p_bar:
                p_bar.set_description(f"[{evaluation}] epoch {epoch}/{num_epochs} | batch {idx}/{train_batches}",
                                      refresh=True)

    y_pred_tmp = torch.concatenate(y_pred_tmp).cpu().numpy()
    y_true_tmp = torch.concatenate(y_true_tmp).cpu().numpy()

    loss_tmp = torch.stack(loss_tmp)
    metrics[f"{evaluation[:5]}_loss"] = loss_tmp.mean().item()
    metrics[f"{evaluation[:5]}_loss_std"] = loss_tmp.std().item()
    metrics[f"{evaluation[:5]}_f1"] = sklearn.metrics.f1_score(y_true=y_true_tmp, y_pred=y_pred_tmp,
                                                               average='weighted')
    metrics[f"{evaluation[:5]}_acc"] = sklearn.metrics.accuracy_score(y_true=y_true_tmp, y_pred=y_pred_tmp)


def train_classifier(model: nn.Module,
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
                     current_state:Dict):

    train_dl, valid_dl = dataset

    es = EarlyStopping(patience if patience > 0 else num_epochs)

    model.to(device)
    criterion = CrossEntropyLoss().to(device)
    optimizer = optimizer(model.parameters(), **opt_params)

    if scheduler:
        scheduler = scheduler(optimizer, **sc_params)

    train_batches = len(train_dl)
    valid_batches = len(valid_dl)

    p_bar = tqdm(range(num_epochs), desc="[Training]")
    p_bar.refresh()

    current_state["time_start"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    for epoch in range(num_epochs):
        if es.earlyStop:
            break

        # ------------------------------------------------------------------------
        model.train()

        p_bar.set_description(f"[Training] epoch {epoch}/{num_epochs}", refresh=True)
        current_state["epoch"] = f"[Training] epoch {epoch}/{num_epochs}"
        for idx, (x, y_true) in enumerate(train_dl, 1):
            x, y_true = x.to(device), y_true.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss_tensor = criterion(y_pred, y_true)
            loss_tensor.backward()
            optimizer.step()

            p_bar.set_description(f"[Training] epoch {epoch}/{num_epochs} | batch {idx}/{train_batches}", refresh=True)
            current_state["batch"] = f"Batch {idx}/{train_batches}"
        # ------------------------------------------------------------------------
        model.eval()

        p_bar.set_description(f"[Validation] epoch {epoch}/{num_epochs}", refresh=True)
        current_state["epoch"] = f"[Validation] epoch {epoch}/{num_epochs}"

        metrics = dict()
        eval_dataset(train_dl, model, criterion, metrics, device, "training", p_bar, epoch, num_epochs, train_batches)
        eval_dataset(valid_dl, model, criterion, metrics, device, "validation", p_bar, epoch, num_epochs, valid_batches)
        # ------------------------------------------------------------------------

        update_cache(metrics=metrics, output_file=metric_history)
        torch.save(model.cpu().state_dict(), Path(model_cache))

        # Update the early stopping
        es.update(metrics["valid_f1"] * -1)

        p_bar.set_postfix(tr_loss=metrics["train_loss"], vl_loss=metrics["valid_loss"], tr_f1=metrics["train_f1"],
                          vl_f1=metrics["valid_f1"])
        p_bar.update()

        current_state["tr_loss_mean"] = metrics["train_loss"]
        current_state["tr_loss_std"] = metrics["train_loss_std"]
        current_state["vl_loss_mean"] = metrics["valid_loss"]
        current_state["vl_loss_std"] = metrics["valid_loss_std"]
        current_state["vl_f1"] = metrics["train_f1"]
        current_state["vl_acc"] = metrics["valid_f1"]
        current_state["time_elapsed"] =  datetime.datetime.fromtimestamp(time.time()) - current_state["time_start"]
    return