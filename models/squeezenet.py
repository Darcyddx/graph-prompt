"""squeezenet in pytorch



[1] Song Han, Jeff Pool, John Tran, William J. Dally

    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mask import RelationshipLayer

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=10):

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.relationship_layer = RelationshipLayer(similarity='cos')

    def forward(self, x, state):
        relation_lists = []  # 12+1

        x = self.stem(x)
        if state[0]:
            r1 = self.relationship_layer(x, state[0])
            relation_lists.append(r1)

        f2 = self.fire2(x)
        if state[1]:
            r2 = self.relationship_layer(f2, state[1])
            relation_lists.append(r2)

        f3 = self.fire3(f2) + f2
        if state[2]:
            r3 = self.relationship_layer(f3, state[2])
            relation_lists.append(r3)

        f4 = self.fire4(f3)
        if state[3]:
            r4 = self.relationship_layer(f4, state[3])
            relation_lists.append(r4)

        f4 = self.maxpool(f4)
        if state[4]:
            r5 = self.relationship_layer(f4, state[4])
            relation_lists.append(r5)

        f5 = self.fire5(f4) + f4
        if state[5]:
            r6 = self.relationship_layer(f5, state[5])
            relation_lists.append(r6)

        f6 = self.fire6(f5)
        if state[6]:
            r7 = self.relationship_layer(f6, state[6])
            relation_lists.append(r7)

        f7 = self.fire7(f6) + f6
        if state[7]:
            r8 = self.relationship_layer(f7, state[7])
            relation_lists.append(r8)

        f8 = self.fire8(f7)
        if state[8]:
            r9 = self.relationship_layer(f8, state[8])
            relation_lists.append(r9)

        f8 = self.maxpool(f8)
        if state[9]:
            r10 = self.relationship_layer(f8, state[9])
            relation_lists.append(r10)

        f9 = self.fire9(f8)
        if state[10]:
            r11 = self.relationship_layer(f8, state[10])
            relation_lists.append(r11)

        c10 = self.conv10(f9)
        if state[11]:
            r12 = self.relationship_layer(f8, state[11])
            relation_lists.append(r12)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        output_softmax = F.softmax(x, dim=1)

        if state[12]:
            r13 = self.relationship_layer(output_softmax, state[12])
            relation_lists.append(r13)

        return x, relation_lists

def squeezenet(class_num=10):
    return SqueezeNet(class_num=class_num)
