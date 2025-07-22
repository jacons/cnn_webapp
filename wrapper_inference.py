from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from torchvision.transforms import transforms as T
from PIL.Image import Image

from model.resnet_model import CNNClassifier
from utils import load_json


class WebInference:
    def __init__(self):
        self.model_dir = None
        self.model = None
        self.params = None
        self.idx2class = None

        self.tfms = T.Compose([
            T.Resize((400, 400)),  # Resize images to 400x400 pixels
            T.ToTensor(),          # Convert PIL Image to PyTorch Tensor
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize image pixel values
        ])

    def prepare_img(self, img: Image)-> torch.Tensor:
        return self.tfms(img).unsqueeze(0)

    def inference(self, img:Image, device: str, model_name: str)-> DataFrame:

        if not self.model or self.model_dir != model_name:
            self.model_dir = model_name
            model_name = Path(model_name)
            self.params = load_json(model_name / "params.json")
            self.idx2class = load_json(model_name / "idx2class.json")

            cls = CNNClassifier(num_classes=self.params["num_classes"],
                                pretrained=self.params["pretrained"],
                                model_name="resnet18")
            cls.load_state_dict(torch.load(model_name / "model.pth", weights_only=True))
            cls.eval()
            self.model = cls.to(device)

        img = self.prepare_img(img).to(device)
        with torch.no_grad():
            logits = self.model(img)

        distr_prob = DataFrame(
            data=torch.softmax(logits, dim=1).cpu().numpy(),
            columns=list(self.idx2class.values()),
            index=["prob"]
        ).T.reset_index().sort_values(by="prob", ascending=True)
        print(distr_prob)
        return distr_prob