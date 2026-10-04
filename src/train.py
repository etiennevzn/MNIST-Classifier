import torch
import torch.nn as nn
import torch.optim as optim
import time

class Trainer:
    def __init__(self, model, train_loader, learning_rate, epochs):
        self.model = model
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.epochs):
            loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss : {loss:.4f}")

    def train_epoch(self):
        epoch_loss = 0
        batches = 0

        for images, labels in self.train_loader:
            batches += 1

            self.optimizer.zero_grad()

            y_train = self.model(images)
            loss = self.criterion(y_train, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / batches
    
        