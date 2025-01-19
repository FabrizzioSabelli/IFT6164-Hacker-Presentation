"""
Substitue models for MNIST dataset
"""

import torch

import torch.nn as nn
import torch
from typing import Dict
from torch import Tensor
import torch.nn.functional as F
from oracle import DNNOracle
from augmented_dataset import AugmentedDataset


class DNNSubstituteSameOracle(DNNOracle):
    # Architecture A in the paper of Papernot et al.

    def __init__(
        self,
    ) -> None:
        super(DNNSubstituteSameOracle, self).__init__()


# TODO change the architecture
class DNNSubstituteDiffOracle(nn.Module):
    """Subtitute DNN with architecture being different as the Oracle."""

    # Architecture ___ in the paper of Papernot et al.
    # Architecture ___ in the paper of Papernot et al.

    def __init__(
        self,
    ) -> None:
        super(DNNSubstituteSameOracle, self).__init__()

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
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.ff_layers(x)
        out = F.log_softmax(x, dim=1)
        return out
