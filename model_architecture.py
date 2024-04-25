import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # MNIST images are 28x28
            nn.ReLU(),
            nn.Linear(512, 10)  # 10 classes for MNIST digits
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)