"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from conf import settings

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100, preprocessing={}):
        super().__init__()
        self.features = features

        self.preprocessing = preprocessing
        
        if not len(self.preprocessing) == 0:
            self.mu = torch.tensor(preprocessing['mean']).float().view(3, 1, 1).cuda()
            self.sigma = torch.tensor(preprocessing['std']).float().view(3, 1, 1).cuda()

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        if not len(self.preprocessing) == 0:
            x = (x - self.mu) / self.sigma
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(num_class=100):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_class)

def vgg13_bn(num_class=100):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_class)

def vgg16_bn(num_class=100, preprocessing={}):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_class, preprocessing=preprocessing)

def vgg19_bn(num_class=100):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_class)
