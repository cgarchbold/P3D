import math

import torch
import torch.nn as nn
import torch.nn.init as init

"""
Adapted from: https://github.com/w86763777/pytorch-gan-collections/blob/master/source/models/wgangp.py

"""
class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    
class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResGenerator32(nn.Module):
    def __init__(self, z_dim, num_classes=10, label_embed_size = 5, multi_label = False, num_labels = 1):
        super().__init__()
        self.z_dim = z_dim

        if num_classes != None:
            if multi_label:
                self.multi_label = True
                self.conditional = True
                self.num_labels = num_labels
                self.linear = nn.Linear(z_dim + num_labels, 4 * 4 * 256)
            else:
                self.label_embedding = nn.Embedding(num_classes, label_embed_size)
                self.conditional = True
                self.multi_label = False
                self.linear = nn.Linear(z_dim + label_embed_size, 4 * 4 * 256)
        else:
            self.linear = nn.Linear(z_dim, 4 * 4 * 256)
            self.conditional = False

        self.blocks = nn.Sequential(
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z, label = None):

        if self.conditional:
            if self.multi_label:
                #label = label.reshape([label.shape[0], self.num_labels, 1,1])
                z = torch.cat((z, label), dim=1)
                z = self.linear(z)
            else:
                label_embed = self.label_embedding(label)
                z = torch.cat((z, label_embed), dim=1)
                z = self.linear(z)
        else:    
            z = self.linear(z)

        z = z.view(-1, 256, 4, 4)
        return self.output(self.blocks(z))

class ResDiscriminator32(nn.Module):
    def __init__(self, num_classes = 10, multi_label = False, num_labels = 1):
        super().__init__()
        self.image_size = 32

        if num_classes != None:
            if multi_label:
                self.multi_label = True
                self.conditional = True
                self.num_labels = num_labels
                self.inp = OptimizedResDisblock(3+num_labels, 128)
            else:
                self.label_embedding = nn.Embedding(num_classes, self.image_size*self.image_size)
                self.conditional = True
                self.inp = OptimizedResDisblock(3+1, 128)
        else:
            self.inp = OptimizedResDisblock(3, 128)
            self.conditional = False

        self.model = nn.Sequential(
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = nn.Linear(128, 1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x, label = None):

        if self.conditional:
            if self.multi_label:
                label = label.reshape([label.shape[0],self.num_labels,1,1])
                #expand to image dimension
                label = label.expand([label.shape[0],self.num_labels,self.image_size,self.image_size])
                x = torch.cat((x, label), dim=1)
            else:
                label_embed = self.label_embedding(label)
                label_embed = label_embed.reshape([label_embed.shape[0], 1, self.image_size, self.image_size])
                x = torch.cat((x, label_embed), dim=1)
                
        x = self.inp(x)
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x