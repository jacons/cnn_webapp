import json
import os
import random
import time
from pathlib import Path
from threading import Thread
from typing import Type

from flask import Flask, render_template, request, jsonify, redirect
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


current_state = {"training": False, 'status': 0, "tr_loss_mean":0, "tr_loss_std":0,
            "vl_loss_mean":0, "vl_loss_std":0, "vl_f1":0, "vl_acc":0,
            "time_start":0, "time_elapsed":0, "epoch":0, "batch":0}

training_thread = None
dataset_list = [nome for nome in os.listdir(DATASET_FOLDER)  if os.path.isdir(os.path.join(DATASET_FOLDER, nome))]
gpu_list = get_available_accelerators()


def train_in_background(params:dict):
    global current_state

    if current_state['training']:
        return
    current_state['training'] = True

    train_dt = CustomDataset(data_path=Path("./datasets/" + params["dataset_name"]), portion="train")
    train_loader = DataLoader(train_dt, batch_size=params["batch_size"], shuffle=True, num_workers=4)

    valid_dt = CustomDataset(data_path=Path("./datasets/"+ params["dataset_name"]), portion="valid")
    valid_loader = DataLoader(valid_dt, batch_size=params["batch_size"], shuffle=True, num_workers=4)

    cls = CNNClassifier(num_classes=params["num_classes"], pretrained=params["pretrained"])

    (Path("./saved_models") / params["output_dir"]).mkdir(parents=True, exist_ok=True)
    metric_history = Path("./saved_models") / params["output_dir"] / "history.json"
    with metric_history.open("w") as file:
        json.dump([], file)

    train_classifier(
        model=cls,
        dataset=(train_loader, valid_loader),
        optimizer=params["optimizer"],
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        num_epochs=params["num_epochs"],
        opt_params=dict(lr=params["lr"]),
        sc_params=dict(mode="max", patience=3, threshold=0.9),
        metric_history=metric_history,
        model_cache="./saved_models/" + params["output_dir"] + "/model.pth",
        patience=params["patience"],
        device=params["device"],
        current_state=current_state,
    )

    current_state['status'] = 100
    current_state['training'] = False


def get_optimizer_from_str(opt_text)-> Type[optim.Optimizer]:
    if opt_text == "Adam":
        return optim.Adam
    elif opt_text == "SGD":
        return optim.SGD
    elif opt_text == "L-BFGS":
        return optim.LBFGS
    elif opt_text == "Averaged Stochastic Gradient Descent":
        return optim.ASGD
    else:
        raise ValueError("Invalid optimizer name")

@app.route('/', methods=['GET', 'POST'])
def index():
    print("inxex")
    return render_template('index.html', dataset_list=dataset_list, gpu_list=gpu_list)

@app.route('/training', methods=['GET', 'POST'])
def training_phase():
    global training_thread

    if request.method == 'POST':
        print(request.form)

        action = request.form['action']

        if action == 'training':
            params = dict()
            params["num_classes"] = int(request.form['num_classes'])
            params["dataset_name"] = str(request.form['dataset_name'])
            params["pretrained"] = request.form['pretrained'] == "True"
            params["optimizer"] = get_optimizer_from_str(str(request.form['optimizer']))
            params["lr"] = float(request.form['lr'])
            params["batch_size"] = int(request.form['batch_size'])
            params["num_epochs"] = int(request.form['epochs'])
            params["patience"] = int(request.form['early_stopping_patience'])
            params["gpu_engine"] = str(request.form['gpu_engine'])
            params["output_dir"] = str(request.form['output_folder_name'])
            params["device"] = params["gpu_engine"]

            training_thread = Thread(target=train_in_background, args=(params,))
            training_thread.start()

    return redirect('index.html')
@app.route('/progress')
def get_progress():
    return jsonify(current_state)


if __name__ == '__main__':
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    app.run(port=9000, debug=True)
