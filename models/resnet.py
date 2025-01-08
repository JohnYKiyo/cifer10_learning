import torch.nn as nn
import torchvision.models as models


def get_resnet(num_classes=10):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 出力層をCIFAR-10用に変更
    return model
