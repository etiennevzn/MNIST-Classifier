import torch
import torch.nn as nn
import torch.optim as optim

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
            self.train_epoch()

    def train_epoch(self):
        for images, labels in self.train_loader:
            self.optimizer.zero_grad()
            y_train = self.model(images)
            loss = self.criterion(y_train, labels)
            loss.backward()
            self.optimizer.step()

    
        