"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gcr import RelationshipLayer


class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=10):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

        self.relationship_layer = RelationshipLayer(similarity='cos')

    def forward(self, x, state):
        output = []  # 15 + 1

        x = self.prelayer(x)
        if state[0]:
            x1 = self.relationship_layer(x, state[0])
            output.append(x1)

        x = self.maxpool(x)
        if state[1]:
            x2 = self.relationship_layer(x, state[1])
            output.append(x2)

        x = self.a3(x)
        if state[2]:
            x3 = self.relationship_layer(x, state[2])
            output.append(x3)

        x = self.b3(x)
        if state[3]:
            x4 = self.relationship_layer(x, state[3])
            output.append(x4)

        x = self.maxpool(x)
        if state[4]:
            x5 = self.relationship_layer(x, state[4])
            output.append(x5)

        x = self.a4(x)
        if state[5]:
            x6 = self.relationship_layer(x, state[5])
            output.append(x6)

        x = self.b4(x)
        if state[6]:
            x7 = self.relationship_layer(x, state[6])
            output.append(x7)

        x = self.c4(x)
        if state[7]:
            x8 = self.relationship_layer(x, state[7])
            output.append(x8)

        x = self.d4(x)
        if state[8]:
            x9 = self.relationship_layer(x, state[8])
            output.append(x9)

        x = self.e4(x)
        if state[9]:
            x10 = self.relationship_layer(x, state[9])
            output.append(x10)

        x = self.maxpool(x)
        if state[10]:
            x11 = self.relationship_layer(x, state[10])
            output.append(x11)

        x = self.a5(x)
        if state[11]:
            x12 = self.relationship_layer(x, state[11])
            output.append(x12)

        x = self.b5(x)
        if state[12]:
            x13 = self.relationship_layer(x, state[12])
            output.append(x13)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        if state[13]:
            x14 = self.relationship_layer(x, state[13])
            output.append(x14)

        x = self.dropout(x)
        if state[14]:
            x15 = self.relationship_layer(x, state[14])
            output.append(x15)

        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        output_softmax = F.softmax(x, dim=1)

        if state[15]:
            x16 = self.relationship_layer(output_softmax, state[15])
            output.append(x16)

        # Return original output without softmax, graph list, and output graph
        return x, output

def googlenet():
    return GoogleNet()


