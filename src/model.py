import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self, neurons, hidden_layers = 2):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.layers = nn.ModuleList()
        self.initLayers()

    def initLayers(self):
        for i in range(self.hidden_layers):
            if(i == 0):
                self.layers.append(nn.Linear(784, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))

        self.layers.append(nn.Linear(self.neurons, 10))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for (idx, layer) in enumerate(self.layers):
            x = layer(x)
            if(idx < len(self.layers) - 1) : x = nn.functional.relu(x)

        return x

        