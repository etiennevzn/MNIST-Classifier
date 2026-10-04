import torch
import os
from src.data_downloader import DataDownloader
from src.model import MNISTClassifier
from src.train import Trainer
from torch.utils.data import DataLoader

if __name__ == "__main__":
    downloader = DataDownloader()
    train_dataset, test_dataset = downloader.download_mnist()
    
    model = MNISTClassifier(128, 2)
    model_path = "mnist_model.pth"

    if os.path.exists(model_path):
        print("Pretrained model parameters file found. Loading model...")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded!")
    else:
        print("No pretrained model found. Training...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            learning_rate=0.001,
            epochs=10
        )

        trainer.train()

        print("Training over. Saving model...")
        torch.save(model.state_dict(), model_path)
        print("Model saved!")

