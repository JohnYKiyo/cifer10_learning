import os

from data.dataloader import get_cifer10_testloader, get_cifer10_trainloader
from models.simple_cnn import SimpleCNN
from training.trainer import Trainer
from utils.device import get_device


def main():
    # データローダーの作成
    train_loader = get_cifer10_trainloader()
    val_loader = get_cifer10_testloader()

    # モデルの作成
    model = SimpleCNN()

    # デバイスの取得
    device = get_device()
    print(f"Using device: {device}")

    # トレーナーの作成
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        device=device,
    )

    # 学習パラメータ
    epochs = 20
    best_val_loss = float("inf")  # 初期値として無限大
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)  # 保存ディレクトリ作成

    # 学習ループ
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # 1エポックの学習と検証
        train_loss, train_acc = trainer.train_one_epoch()
        val_loss, val_acc = trainer.validate()

        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save(checkpoint_dir, epoch)  # トレーナーを使って保存
            print(f"Best model updated with Validation Loss: {val_loss:.4f}")

        # エポック終了時のログ出力
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    print(
        f"\nTraining complete. Best model saved at: {os.path.join(checkpoint_dir, 'checkpoint.pt')}"
    )


if __name__ == "__main__":
    main()
