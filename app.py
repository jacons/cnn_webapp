import argparse
import base64
import io
import os
from pathlib import Path
from threading import Thread
from typing import Type

import matplotlib
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, url_for
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

from ml_utils import CustomDataset, get_available_accelerators, get_annotates_classes
from model.resnet_model import CNNClassifier
from training import train_classifier
from utils import find_folders_with_model, save_json
from wrapper_inference import WebInference

matplotlib.use('Agg')
# ---------------- PARAMETERS AND CONSTANT ----------------
# ---------------- PARAMETERS AND CONSTANT ----------------
DATASET_FOLDER = 'datasets'
MODEL_FOLDER = 'webapp_result'
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
global mode

inference_model = WebInference()

current_state = {
    "training": False,  # if training or validation
    "status": 0,  # 0-100 percentage of training progress
    "tr_loss_mean": 0,  # mean training loss
    "tr_loss_std": 0,  # std training loss
    "vl_loss_mean": 0,  # mean validation loss
    "vl_loss_std": 0,  # std validation loss
    "vl_f1": 0,  # validation F1 score
    "vl_acc": 0,  # validation accuracy
    "time_start": 0,  # time when training started
    "time_elapsed": 0,  # time elapsed since training started
    "epoch": 0,  # current epoch
    "batch": 0  # current batch
}

training_thread = None

# Get the list of datasets and available GPUs
dataset_list = [name for name in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, name))]
model_list = find_folders_with_model(MODEL_FOLDER)

gpu_list = get_available_accelerators()

# Define a dict of possible optimizers
optimizer_map = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "L-BFGS": optim.LBFGS,
    "Averaged Stochastic Gradient Descent": optim.ASGD,
}


# ---------------- PARAMETERS AND CONSTANT ----------------
# ---------------- PARAMETERS AND CONSTANT ----------------

# ---------------- FUNCTIONS ----------------

def get_optimizer_from_str(opt_text: str) -> Type[optim.Optimizer]:
    """ Convert a string representation of an optimizer to the corresponding PyTorch optimizer class."""
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
    idx2class = get_annotates_classes(dataset_path)
    model = CNNClassifier(num_classes=params["num_classes"], pretrained=params["pretrained"])
    metric_history_path = model_path / "history.json"
    metric_history_path.write_text("[]")

    train_classifier(
        model=model,
        dataset=(train_loader, valid_loader),
        optimizer_cls=params["optimizer"],
        opt_params={"lr": params["lr"]},
        scheduler_cls=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params=dict(T_max=params["num_epochs"], eta_min=1e-6),
        num_epochs=params["num_epochs"],
        metric_history=metric_history_path,
        model_cache=model_path / "model.pth",
        patience=params["patience"],
        device=params["device"],
        state=current_state,
    )

    current_state.update({"status": 100, "training": False})
    save_json(params, model_path / "params.json", default=str)
    save_json(idx2class, model_path / "idx2class.json")
# ---------------- FUNCTIONS ----------------


app = Flask(__name__)

app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    if mode == "train":
        return render_template('index.html', dataset_list=dataset_list, gpu_list=gpu_list)
    elif mode == "inference":
        return render_template('inference.html', model_list=model_list, gpu_list=gpu_list)
    return None

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
            "output_dir": request.form['output_folder_name'],
            "device": request.form['gpu_engine'],
        }
        training_thread = Thread(target=train_in_background, args=(params,))
        training_thread.start()

    return redirect(url_for('index'))

@app.route('/inference', methods=['POST'])
def get_inference_phase():

    if 'img_input' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['img_input']

    # Check if the file is empty
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if image_file.mimetype not in ["image/png", "image/jpeg", "image/jpg"]:
        return jsonify({"error": f"Invalid file type. Please upload a PNG or JPEG image. get: {image_file.mimetype}"}), 400

    if not image_file:
        return jsonify({'error': 'No image file provided'}), 400

    device = request.form['gpu_engine']
    model_name = request.form['model_name']
    pil_image = Image.open(image_file.stream).convert('RGB')  # Ensure RGB for consistency

    dist = inference_model.inference(img=pil_image,device=device, model_name=model_name)

    f, axs = plt.subplots(nrows=1, ncols=2,figsize=(15, 10))

    plt.suptitle(f"Prediction: {dist.tail(1)["index"].values[0]}", fontsize=20, fontweight='bold', color="#E69F00")
    axs[0].imshow(pil_image)
    axs[0].axis('off')
    axs[0].set_title('Image', fontsize=15, fontweight='bold')

    axs[1].set_title('Top 15 Predictions', fontsize=15, fontweight='bold')
    axs[1].barh(dist["index"][::15], dist["prob"][::15], height=0.5, color="#53b298", edgecolor="#009E73")
    axs[1].set_axisbelow(True)
    axs[1].yaxis.grid(color='gray', linestyle='-')
    axs[1].xaxis.grid(color='gray', linestyle='-')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return render_template('inference.html', model_list=model_list, gpu_list=gpu_list,
                           result=image_base64)

@app.route('/progress')
def get_progress():
    return jsonify(current_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a car classifier model.")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train mode"
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Inference mode"
    )
    args = parser.parse_args()

    mode = "train" if args.train else "inference"

    app.run(port=9355, debug=True)