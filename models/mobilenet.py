import torch
import torch.nn as nn
import torch.nn.functional as F

from mask import RelationshipLayer


class DepthSeperabelConv2d(nn.Module):
    # This class remains unchanged from the original mobilenet.py
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BasicConv2d(nn.Module):
    # This class remains unchanged from the original mobilenet.py
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, width_multiplier=1, class_num=10):
        super().__init__()

        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(
                int(32 * alpha),
                int(64 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv1 = nn.Sequential(
            DepthSeperabelConv2d(
                int(64 * alpha),
                int(128 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(128 * alpha),
                int(128 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv2 = nn.Sequential(
            DepthSeperabelConv2d(
                int(128 * alpha),
                int(256 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(256 * alpha),
                int(256 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv3 = nn.Sequential(
            DepthSeperabelConv2d(
                int(256 * alpha),
                int(512 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv4 = nn.Sequential(
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(1024 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(1024 * alpha),
                int(1024 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        self.fc = nn.Linear(int(1024 * alpha), class_num)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.relationship_layer = RelationshipLayer(similarity='cos')

    def forward(self, x, state):
        output = []  # 6+1

        # Stem
        x = self.stem(x)
        if state[0]:
            x1 = self.relationship_layer(x, state[0])
            output.append(x1)

        # Conv1
        x = self.conv1(x)
        if state[1]:
            x2 = self.relationship_layer(x, state[1])
            output.append(x2)

        # Conv2
        x = self.conv2(x)
        if state[2]:
            x3 = self.relationship_layer(x, state[2])
            output.append(x3)

        # Conv3
        x = self.conv3(x)
        if state[3]:
            x4 = self.relationship_layer(x, state[3])
            output.append(x4)

        # Conv4
        x = self.conv4(x)
        if state[4]:
            x5 = self.relationship_layer(x, state[4])
            output.append(x5)

        # Average pooling and fully connected layer
        x = self.avg(x)
        # Calculate graph if state_list[5] is 1
        if state[5]:
            x6 = self.relationship_layer(x, state[5])
            output.append(x6)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Apply softmax for graph calculation
        output_softmax = F.softmax(x, dim=1)

        if state[6]:
            x7 = self.relationship_layer(output_softmax, state[6])
            output.append(x7)

        # Return original output without softmax, graph list, and output graph
        return x, output


def mobilenet(alpha=1, class_num=10):
    return MobileNet(alpha, class_num)