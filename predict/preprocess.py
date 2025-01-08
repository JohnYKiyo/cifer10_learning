import torch
from torchvision import transforms


def preprocess_image(image):
    """
    画像データを前処理してテンソルに変換する
    Args:
        image: PIL.Image, numpy.ndarray, または torch.Tensor
    Returns:
        tensor: モデルに入力可能なテンソル
    """
    if isinstance(image, torch.Tensor):
        # 入力がすでに Tensor の場合はそのまま返す
        return image.unsqueeze(0)

    preprocess = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # CIFAR-10 のサイズに合わせる
            transforms.ToTensor(),
        ]
    )
    return preprocess(image).unsqueeze(0)  # バッチ次元を追加
