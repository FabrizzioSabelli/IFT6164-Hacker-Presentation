"""
Oracle models for MNIST dataset
"""

import torch
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import Dict
from torch import Tensor


class DNNOracle(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(DNNOracle, self).__init__()

        # Convolutational layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.ff_layers = nn.Sequential(
            nn.Linear(24 * 24 * 64, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_layers(x)
        out = torch.flatten(out, 1)
        out = self.ff_layers(out)
        return out
