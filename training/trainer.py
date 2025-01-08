import os

import torch
import torch.nn as nn
from schedulefree import RAdamScheduleFree
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device="cpu",
        lr=0.001,
        batas=(0.9, 0.999),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  # 損失関数
        self.optimizer = RAdamScheduleFree(self.model.parameters(), lr=lr, betas=batas)

    def train_one_epoch(self):
        self.model.train()
        self.optimizer.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        self.optimizer.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss, accuracy

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    def save(self, save_dir: str, epoch: int):
        """
        モデルとオプティマイザの状態を保存します。

        Args:
            save_dir (str): 保存先のディレクトリパス。
            epoch (int): 現在のエポック数を保存。
        """
        # 保存ディレクトリが存在しない場合は作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "checkpoint.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            save_path,
        )
        print(f"Checkpoint saved at {save_path}")
