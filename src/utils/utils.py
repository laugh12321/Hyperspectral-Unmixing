#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: utils.py
@desc: Helper functions.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import numpy as np
from typing import Dict, List, Union

from src.model import enums

SAMPLES_DIM = 0
MEAN_PER_CLASS_ACC = 'mean_per_class_accuracy'


def subsample_test_set(data: Dict, test_size: int) -> None:
    """
    Subsample the test set.

    :param data: The test data dictionary.
    :param test_size: Size of the test samples to remain.
    """
    assert data[enums.Dataset.DATA].shape[SAMPLES_DIM] >= test_size and \
           data[enums.Dataset.LABELS].shape[SAMPLES_DIM] >= test_size
    data[enums.Dataset.DATA] = data[enums.Dataset.DATA][-test_size:]
    data[enums.Dataset.LABELS] = data[enums.Dataset.LABELS][-test_size:]


def shuffle_arrays_together(arrays: List[np.ndarray], seed: int = 0) -> None:
    """
    Shuffle arbitrary number of arrays together, in-place

    :param arrays: List of np.ndarrays to be shuffled
    :param seed: seed for the random state, defaults to 0
    :raises AssertionError: When provided arrays have different sizes along 
                            first dimension
    """
    assert all(len(array) == len(arrays[0]) for array in arrays)
    for array in arrays:
        random_state = np.random.RandomState(seed)
        random_state.shuffle(array)


def _build_data_dict(train_x, train_y, val_x, val_y, test_x, test_y) -> Dict:
    """
    Build data dictionary with following structure:

    'train':
        'data': np.ndarray
        'labels': np.ndarray
    'val':
        'data': np.ndarray
        'labels': np.ndarray
    'test':
        'data': np.ndarray
        'labels' np.ndarray
    'min': float
    'max': float

    :param train_x: Train set
    :param train_y: Train labels
    :param val_x: Validation set
    :param val_y: Validation labels
    :param test_x: Test set
    :param test_y: Test labels
    :return: Dictionary containing train, validation and test subsets.
    """
    data_dict = {}
    train_min, train_max = np.amin(train_x), np.amax(train_x)
    data_dict[enums.DataStats.MIN] = train_min
    data_dict[enums.DataStats.MAX] = train_max

    data_dict[enums.Dataset.TRAIN] = {}
    data_dict[enums.Dataset.TRAIN][enums.Dataset.DATA] = train_x
    data_dict[enums.Dataset.TRAIN][enums.Dataset.LABELS] = train_y

    data_dict[enums.Dataset.VAL] = {}
    data_dict[enums.Dataset.VAL][enums.Dataset.DATA] = val_x
    data_dict[enums.Dataset.VAL][enums.Dataset.LABELS] = val_y

    data_dict[enums.Dataset.TEST] = {}
    data_dict[enums.Dataset.TEST][enums.Dataset.DATA] = test_x
    data_dict[enums.Dataset.TEST][enums.Dataset.LABELS] = test_y
    return data_dict


def restructure_per_class_accuracy(metrics: Dict[str, List[float]]) -> Dict[
    str, List[float]]:
    """
    Restructure mean accuracy values of each class under
    'mean_per_class_accuracy' key, to where each class' accuracy value lays
    under it's specific key
    :param metrics: Dictionary with metric names and corresponding values
    :return: Dictionary with modified per class accuracy
    """
    if MEAN_PER_CLASS_ACC in metrics.keys():
        per_class_acc = {'Class_' + str(i):
                             [item] for i, item in
                         enumerate(*metrics[MEAN_PER_CLASS_ACC])}
        metrics.update(per_class_acc)
        del metrics[MEAN_PER_CLASS_ACC]
    return metrics


def get_central_pixel_spectrum(data: np.ndarray,
                               neighborhood_size: int) -> np.ndarray:
    """
    If the model is an autoencoder, get the central pixel
    spectrum as its reconstruction original sample.

    :param data: The data cube used for training the autoencoder.
    :param neighborhood_size: Spatial size of the patch.
    """
    if neighborhood_size is not None:
        central_index = np.floor(neighborhood_size / 2).astype(int)
        return np.squeeze(data[:, :, central_index, central_index])
    return np.squeeze(data)


def parse_train_size(train_size: List) -> Union[float, int, List[int]]:
    """
    If single element list provided, convert to int or float based on provided
    value
    :param train_size: Train size as list
    :return: Converted type
    """
    if type(train_size) is not list:
        return train_size
    if len(train_size) == 1:
        train_size = float(train_size[0])
        if 0.0 <= train_size <= 1:
            return float(train_size)
        else:
            return int(train_size)
    else:
        return list(map(int, train_size))


def get_label_indices_per_class(labels, return_uniques: bool = True):
    """
    Extract indices of each class
    :param labels: Data labels
    :param return_uniques: Whether to return unique labels contained in
        labels arg
    :return: List with lists of label indices of consecutive labels
    """
    unique_labels = np.unique(labels)
    label_indices = [np.where(labels == label)[0] for label in unique_labels]
    if return_uniques:
        return label_indices, unique_labels
    else:
        return label_indices
