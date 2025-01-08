import torch

from models.simple_cnn import SimpleCNN
from predict.preprocess import preprocess_image
from utils.device import get_device


class Predictor:
    def __init__(self, model_path: str, num_classes: int = 10):
        self.device = get_device()
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()

    def predict(self, image):
        """
        1枚の画像の予測を行う
        """
        # データ前処理
        tensor_image = preprocess_image(image).to(self.device)
        # 推論
        with torch.no_grad():
            outputs = self.model(tensor_image)
            _, predicted = outputs.max(dim=1)

        return predicted.item()
