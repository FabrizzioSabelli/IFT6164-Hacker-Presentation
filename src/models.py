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
        self.classifier = torch.load(file_path, map_location=self.device)

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

class ResNeXtBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, cardinality=32
    ):
        super(ResNeXtBlock, self).__init__()
        D = out_channels // cardinality  # Dimension of each group

        self.conv1 = nn.Conv2d(
            in_channels, D * cardinality, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(D * cardinality)

        self.conv2 = nn.Conv2d(
            D * cardinality,
            D * cardinality,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(D * cardinality)

        self.conv3 = nn.Conv2d(
            D * cardinality, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, block, layers, num_classes=10, cardinality=32, dropout=0.1):
        super(ResNeXt, self).__init__()
        self.in_channels = 64

        # Initial convolution tailored for CIFAR-10
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], cardinality)
        self.layer2 = self._make_layer(block, 128, layers[1], cardinality, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cardinality, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cardinality, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample, cardinality)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, cardinality=cardinality)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x


def resnext18_cifar10(num_classes=10, cardinality=32):
    return ResNeXt(
        ResNeXtBlock, [2, 2, 2, 2], num_classes=num_classes, cardinality=cardinality
    )