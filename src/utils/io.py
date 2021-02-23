#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: io.py
@desc: All I/O related functions
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import os
import csv
import glob
import numpy as np
from typing import Dict, List, Tuple

import src.model.enums as enums

UNMIXING_CLASS_FRACTIONS = 0


def load_metrics(experiments_path: str, filename: str = None) -> Dict[
    List, List]:
    """
    Load metrics to a dictionary.

    :param experiments_path: Path to the experiments directory.
    :param filename: Name of the file holding metrics. Defaults to
        'inference_metrics.csv'.
    :return: Dictionary containing all metric names and
        values from all experiments.
    """
    all_metrics = {'metric_keys': [], 'metric_values': []}
    for experiment_dir in glob.glob(
            os.path.join(experiments_path,
                         '{}*'.format(enums.Experiment.EXPERIMENT))):
        if filename is None:
            inference_metrics_path = os.path.join(
                experiment_dir,
                enums.Experiment.INFERENCE_METRICS)
        else:
            inference_metrics_path = os.path.join(experiment_dir, filename)
        with open(inference_metrics_path) as metric_file:
            reader = csv.reader(metric_file, delimiter=',')
            for row, key in zip(reader, all_metrics.keys()):
                all_metrics[key].append(row)
    return all_metrics


def save_metrics(dest_path, metrics, filename=None):
    """
    Save given dataframe of metrics.

    :param dest_path: Destination path.
    :param filename: Name to save the file.
    :param metrics: Dictionary containing all metrics.
    """
    if filename is not None:
        dest_path = os.path.join(dest_path, filename)
    metrics.to_csv(dest_path, index=False, header=True)


def load_npy(data_file_path: str, gt_input_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load .npy data and GT from specified paths

    :param data_file_path: Path to the data .npy file
    :param gt_input_path: Path to the GT .npy file
    :return: Tuple with loaded data and GT
    """
    data, labels = np.load(data_file_path), np.load(gt_input_path)
    height, width, _ = data.shape
    labels = np.moveaxis(labels.reshape(-1, height, width), UNMIXING_CLASS_FRACTIONS, -1)

    return data, labels


def read_min_max(path: str) -> Tuple[float, float]:
    """
    Read min and max value from a .csv file

    :param path: Path to the .csv file containing min and max value
    :return: Tuple with min and max
    """
    min_, max_ = np.loadtxt(path)
    return min_, max_
