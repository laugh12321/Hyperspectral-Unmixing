#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: performance_metrics.py
@desc: All metrics that are calculated on the model's output.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import torch
import numpy as np
from typing import Dict, List

from src.model.models import pixel_based_cnn, cube_based_cnn, pixel_based_dcae


def spectral_information_divergence_loss(y_true, y_pred):
    """
    Calculate the spectral information divergence loss,
    which is based on the divergence in information theory.

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional
    autoencoders in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations and
    Remote Sensing 13 (2020): 567-576.

    :param y_true: Labels as two dimensional abundances or original
        input array of shape: [n_samples, n_classes], [n_samples, n_bands].
    :param y_pred: Predicted abundances or reconstructed input array of shape:
    [n_samples, n_classes], [n_samples, n_bands].
    :return: The spectral information divergence loss.
    """
    y_true_row_sum = torch.sum(y_true, 1)
    y_pred_row_sum = torch.sum(y_pred, 1)
    y_true = y_true / torch.reshape(y_true_row_sum, (-1, 1))
    y_pred = y_pred / torch.reshape(y_pred_row_sum, (-1, 1))
    y_true, y_pred = torch.clamp(y_true, torch.finfo(torch.float32).eps, 1), \
                     torch.clamp(y_pred, torch.finfo(torch.float32).eps, 1)
    loss = torch.sum(y_true * torch.log(y_true / y_pred)) + torch.sum(y_pred * torch.log(y_pred / y_true))
    return loss


def average_angle_spectral_mapper(y_true, y_pred):
    """
    Calculate the dcae average angle spectral mapper value.

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes] or original input pixel
        and its reconstruction of shape: [n_samples, n_bands].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes]
        or original input pixel and
        its reconstruction of shape: [n_samples, n_bands].
    :return: The root-mean square abundance angle distance error.
    """
    numerator = torch.sum(torch.mul(y_true, y_pred), 1)
    y_true_len = torch.sqrt(torch.sum(torch.square(y_true), 1))
    y_pred_len = torch.sqrt(torch.sum(torch.square(y_pred), 1))
    denominator = torch.mul(y_true_len, y_pred_len)
    loss = torch.mean(torch.acos(torch.clamp(numerator / denominator, -1, 1)))

    return loss


def dcae_rmse(y_true, y_pred):
    """
    Calculate the custom dcae root-mean square error,
    which measures the similarity between the original abundance
    fractions and the predicted ones.

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.

    :param y_true: Labels as two dimensional abundances
    array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square error.
    """
    return torch.mean(torch.sqrt(torch.mean(torch.square(y_pred - y_true), 1)))


def overall_rms_abundance_angle_distance(y_true, y_pred):
    """
    Calculate the cnn root-mean square abundance angle distance,
    which measures the similarity between the original abundance fractions
    and the predicted ones. Taken from cnn paper.
    It utilizes the inverse of cosine function at the range [0, pi],
    which means that the domain of arccos is in the range [-1; 1],
    that is why the "tf.clip_by_value" method is used.
    For the identical abundances the numerator / denominator is 1 and
    arccos(1) is 0, which resembles the perfect score.

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square abundance angle distance error.
    """
    numerator = torch.sum(torch.mul(y_true, y_pred))
    y_true_len = torch.sqrt(torch.sum(torch.square(y_true)))
    y_pred_len = torch.sqrt(torch.sum(torch.square(y_pred)))
    denominator = torch.mul(y_true_len, y_pred_len)
    loss = torch.sqrt(torch.mean(torch.square(torch.acos(
        torch.clamp(numerator / denominator, -1, 1)))))
    return loss


def sum_per_class_rmse(y_true, y_pred):
    """
    Calculate the sum of per class root-mean square error.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The sum of per class root-mean square error.
    """
    return torch.mean(per_class_rmse(y_true=y_true, y_pred=y_pred))


def per_class_rmse(y_true, y_pred):
    """
    Calculate the per class root-mean square error vector.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square error vector.
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2, 0))


def cnn_rmse(y_true, y_pred):
    """
    Calculate the custom cnn root-mean square error, which measures the
    similarity between the original abundance fractions and the predicted ones.

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square error.
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


UNMIXING_TRAIN_METRICS = {
    pixel_based_dcae.__name__: [spectral_information_divergence_loss],
    # cube_based_dcae.__name__: [spectral_information_divergence_loss],

    pixel_based_cnn.__name__: [cnn_rmse,
                               overall_rms_abundance_angle_distance,
                               sum_per_class_rmse],
    cube_based_cnn.__name__: [cnn_rmse,
                              overall_rms_abundance_angle_distance,
                              sum_per_class_rmse]
}

UNMIXING_TEST_METRICS = {
    'aRMSE': dcae_rmse,
    'aSAM': average_angle_spectral_mapper,
    'overallRMSE': cnn_rmse,
    'rmsAAD': overall_rms_abundance_angle_distance,
    'perClassSumRMSE': sum_per_class_rmse
}

UNMIXING_LOSSES = {
    pixel_based_dcae.__name__: spectral_information_divergence_loss,
    # cube_based_dcae.__name__: spectral_information_divergence_loss,

    pixel_based_cnn.__name__: torch.nn.MSELoss(),
    cube_based_cnn.__name__: torch.nn.MSELoss()
}


def calculate_unmixing_metrics(**kwargs) -> Dict[str, List[float]]:
    """
    Calculate the metrics for unmixing problem.

    :param kwargs: Additional keyword arguments.
    """
    model_metrics = {}
    for f_name, f_metric in UNMIXING_TEST_METRICS.items():
        model_metrics[f_name] = [float(f_metric(y_true=kwargs['y_true'], y_pred=kwargs['y_pred']))]

    for class_idx, class_rmse in enumerate(per_class_rmse(y_true=kwargs['y_true'],
                                                          y_pred=kwargs['y_pred'])):
        model_metrics[f'class{class_idx}RMSE'] = [float(class_rmse)]
    if kwargs['endmembers'] is not None:
        # Calculate the reconstruction RMSE and SID losses:
        x_pred = torch.matmul(kwargs['y_pred'], kwargs['endmembers'].float())
        model_metrics['rRMSE'] = [dcae_rmse(y_true=kwargs['x_true'],  y_pred=x_pred)]
        model_metrics['rSID'] = [spectral_information_divergence_loss(y_true=kwargs['x_true'], y_pred=x_pred)]
    return model_metrics
