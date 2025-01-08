import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# スクリプトのディレクトリを基準としたデータ保存パスを設定
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "raw")

TRAINPREPROCESS = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

TESTPREPROCESS = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def get_cifer10_trainloader(batch: int = 128, num_workers: int = 2) -> DataLoader:
    trainset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=TRAINPREPROCESS
    )
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=num_workers
    )
    return trainloader


def get_cifer10_testloader(batch: int = 100, num_workers: int = 2) -> DataLoader:
    testset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=TESTPREPROCESS
    )
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers
    )
    return testloader
