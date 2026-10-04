from src.data_downloader import DataDownloader
from src.model import MNISTClassifier
from src.train import Trainer
from torch.utils.data import DataLoader

downloader = DataDownloader()
train_dataset, test_dataset = downloader.download_mnist()
model = MNISTClassifier(128, 2)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
trainer = Trainer(model=model, train_loader=train_loader, learning_rate=0.001, epochs=10)

trainer.train()

