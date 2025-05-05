import os
from pathlib import Path
from threading import Thread
from typing import Type

from flask import Flask, render_template, request, jsonify, redirect, url_for
from torch import optim
from torch.utils.data import DataLoader

from ml_utils import CustomDataset, get_available_accelerators
from model.resnet_model import CNNClassifier
from training import train_classifier

app = Flask(__name__)
DATASET_FOLDER = 'datasets'
MODEL_FOLDER = 'saved_models'
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

current_state = {
    "training": False, "status": 0, "tr_loss_mean": 0, "tr_loss_std": 0,
    "vl_loss_mean": 0, "vl_loss_std": 0, "vl_f1": 0, "vl_acc": 0,
    "time_start": 0, "time_elapsed": 0, "epoch": 0, "batch": 0
}

training_thread = None
dataset_list = [name for name in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, name))]
gpu_list = get_available_accelerators()

optimizer_map = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "L-BFGS": optim.LBFGS,
    "Averaged Stochastic Gradient Descent": optim.ASGD,
}


def get_optimizer_from_str(opt_text: str) -> Type[optim.Optimizer]:
    try:
        return optimizer_map[opt_text]
    except KeyError:
        raise ValueError(f"Invalid optimizer name: {opt_text}")


def train_in_background(params: dict):
    global current_state

    if current_state['training']:
        return

    current_state['training'] = True

    dataset_path = Path(DATASET_FOLDER) / params["dataset_name"]
    model_path = Path(MODEL_FOLDER) / params["output_dir"]
    model_path.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        CustomDataset(data_path=dataset_path, portion="train"),
        batch_size=params["batch_size"], shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        CustomDataset(data_path=dataset_path, portion="valid"),
        batch_size=params["batch_size"], shuffle=True, num_workers=4
    )

    model = CNNClassifier(num_classes=params["num_classes"], pretrained=params["pretrained"])
    metric_history_path = model_path / "history.json"
    metric_history_path.write_text("[]")

    train_classifier(
        model=model,
        dataset=(train_loader, valid_loader),
        optimizer=params["optimizer"],
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        num_epochs=params["num_epochs"],
        opt_params={"lr": params["lr"]},
        sc_params={"mode": "max", "patience": 3, "threshold": 0.9},
        metric_history=metric_history_path,
        model_cache=model_path / "model.pth",
        patience=params["patience"],
        device=params["device"],
        current_state=current_state,
    )

    current_state.update({"status": 100, "training": False})


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', dataset_list=dataset_list, gpu_list=gpu_list)


@app.route('/training', methods=['POST'])
def training_phase():
    global training_thread

    if request.form['action'] == 'training' and not current_state['training']:
        params = {
            "num_classes": int(request.form['num_classes']),
            "dataset_name": request.form['dataset_name'],
            "pretrained": request.form['pretrained'] == "True",
            "optimizer": get_optimizer_from_str(request.form['optimizer']),
            "lr": float(request.form['lr']),
            "batch_size": int(request.form['batch_size']),
            "num_epochs": int(request.form['epochs']),
            "patience": int(request.form['early_stopping_patience']),
            "gpu_engine": request.form['gpu_engine'],
            "output_dir": request.form['output_folder_name'],
            "device": request.form['gpu_engine'],
        }
        training_thread = Thread(target=train_in_background, args=(params,))
        training_thread.start()

    return redirect(url_for('index'))


@app.route('/progress')
def get_progress():
    return jsonify(current_state)


if __name__ == '__main__':
    app.run(port=9000, debug=True)
