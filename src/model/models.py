#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: models.py
@desc: All models that are used in the project.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_model(model_key: str, **kwargs):
    """
    Get a given instance of model specified by model_key.

    :param model_key: Specifies which model to use.
    :param kwargs: Any keyword arguments that the model accepts.
    """
    # Get the list of all model creating functions and their name as the key:
    all_ = {
        str(f): eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](**kwargs)


class Linear(nn.Module):
    """

    """
    def __init__(self, weight, bias):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.from_numpy(weight).float(), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(bias), requires_grad=False)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class pixel_based_cnn(nn.Module):
    """
    Model for pixel-based supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.
    """

    def __init__(self, in_features, n_class, **kwargs):
        super(pixel_based_cnn, self).__init__()
        self.conv1 = nn.Conv3d(1, 3, kernel_size=(1, 1, 5))
        self.conv2 = nn.Conv3d(3, 6, kernel_size=(1, 1, 4))
        self.conv3 = nn.Conv3d(6, 12, kernel_size=(1, 1, 5))
        self.conv4 = nn.Conv3d(12, 24, kernel_size=(1, 1, 4))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=in_features, out_features=192)
        self.fc2 = nn.Linear(in_features=192, out_features=150)
        self.fc3 = nn.Linear(in_features=150, out_features=n_class)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv3(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv4(x)), kernel_size=(1, 1, 2))

        x = F.relu(self.fc1(self.flatten(x)))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x


class cube_based_cnn(nn.Module):
    """
    Model for cube-based supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.
    """

    def __init__(self, in_features, n_class, **kwargs):
        super(cube_based_cnn, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(1, 1, 5))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(1, 1, 4))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(1, 1, 5))
        self.conv4 = nn.Conv3d(64, 128, kernel_size=(1, 1, 4))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=in_features, out_features=192)
        self.fc2 = nn.Linear(in_features=192, out_features=150)
        self.fc3 = nn.Linear(in_features=150, out_features=n_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout3d(F.relu(self.conv3(x)), p=0.2)
        x = F.dropout3d(F.relu(self.conv4(x)), p=0.2)

        x = F.relu(self.fc1(self.flatten(x)))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x


class pixel_based_dcae(nn.Module):
    """
    Model for pixel-based unsupervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.
    """

    def __init__(self, in_features, n_class, **kwargs):
        super(pixel_based_dcae, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=(1, 1, 3))
        self.conv2 = nn.Conv3d(2, 4, kernel_size=(1, 1, 3))
        self.conv3 = nn.Conv3d(4, 8, kernel_size=(1, 1, 3))
        self.conv4 = nn.Conv3d(8, 16, kernel_size=(1, 1, 3))
        self.conv5 = nn.Conv3d(16, 32, kernel_size=(1, 1, 3))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_class)
        self.fc3 = Linear(weight=kwargs['endmembers'], bias=kwargs['input_size'])

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv3(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv4(x)), kernel_size=(1, 1, 2))
        x = F.relu(self.conv5(x))

        x = F.relu(self.fc1(self.flatten(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


class pixel_based_dcae_load(nn.Module):
    """
    Model for pixel-based unsupervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.
    """

    def __init__(self, in_features, n_class, **kwargs):
        super(pixel_based_dcae_load, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=(1, 1, 3))
        self.conv2 = nn.Conv3d(2, 4, kernel_size=(1, 1, 3))
        self.conv3 = nn.Conv3d(4, 8, kernel_size=(1, 1, 3))
        self.conv4 = nn.Conv3d(8, 16, kernel_size=(1, 1, 3))
        self.conv5 = nn.Conv3d(16, 32, kernel_size=(1, 1, 3))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_class)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv3(x)), kernel_size=(1, 1, 2))
        x = F.max_pool3d(F.relu(self.conv4(x)), kernel_size=(1, 1, 2))
        x = F.relu(self.conv5(x))

        x = F.relu(self.fc1(self.flatten(x)))
        x = F.relu(self.fc2(x))

        return x
