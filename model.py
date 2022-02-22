import torch
import torch.nn as nn


class GenderClassifier(nn.Module):

    def __init__(self):
    
        super().__init__()
    
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(3 * 3 * 64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.layers(x)

        return y