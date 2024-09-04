#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#TODO: this module is deprecated, please use resnets/build_model instead

from torch import nn


def get_model(args):
    return CNN4Conv(num_classes=args.num_classes)


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class CNN4Conv(nn.Module):
    def __init__(self, num_classes):
        super(CNN4Conv, self).__init__()
        in_channels = 3
        num_classes = num_classes
        hidden_size = 64

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.linear = nn.Linear(hidden_size * 2 * 2, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        logits = self.linear(features)

        return logits
