import torch

class Evaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

    def evaluate(self):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                y_test = self.model(images)
                _, predicted = torch.max(y_test, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return accuracy
