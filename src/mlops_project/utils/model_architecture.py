import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden1_dim)
        self.layer2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output_layer = nn.Linear(hidden2_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
    