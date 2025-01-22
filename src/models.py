import torch
import torch.nn as nn
from .utils import get_device

class DummyCNN(nn.Module):
    def __init__(
        self,
        conv_filters,
        fc_layers,
        kernel_size=2,
        input_shape=(3, 32, 32),
        num_classes=10,
    ):

        super(DummyCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # Add convolutional blocks
        in_channels = input_shape[0]
        for out_channels in conv_filters:
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.conv_layers.append(block)
            in_channels = out_channels

        # Calculate the size of the feature map after all convolutional blocks
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            for layer in self.conv_layers:
                dummy_input = layer(dummy_input)
            flattened_size = dummy_input.numel()

        # Add fully connected layers
        in_features = flattened_size
        for out_features in fc_layers:
            self.fc_layers.append(
                nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())
            )
            in_features = out_features

        # Add the output layer
        self.fc_layers.append(nn.Linear(in_features, num_classes))
        self.apply(init_weights)

    def forward(self, x):
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the feature map
        x = torch.flatten(x, start_dim=1)

        # Pass through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Oracle(nn.Module):
    def __init__(self, file_path: str):  # device: str = "cuda"):
        super(Oracle, self).__init__()
        # Load the state dictionary
        self.device = get_device()
        self.classifier = torch.load(file_path)

    def forward(self, x):
        if x.device != self.device:
            return self.classifier.forward(x.to(self.device))
        return self.classifier.forward(x)

    def predict(self, image: torch.Tensor):

        self.eval()
        with torch.no_grad():

            # TODO check this
            if len(image.shape) == 3:
                image = image.unsqueeze(dim=0)
            logits = self.forward(
                image
            )  # the function internally moves the image to the device
            return torch.argmax(logits, dim=1)


class Substitute(Oracle):

    def __init__(self, file_path: str):
        super().__init__(file_path)
