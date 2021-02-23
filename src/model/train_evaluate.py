#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: train_evaluate.py
@desc: Perform the training of the models for the unmixing problem.
       Perform the inference of the unmixing models on the test datasets.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import os
import torch
import torchkeras
import numpy as np
import pandas as pd
from typing import Dict
import src.model.enums as enums
from src.utils import io, transforms
from src.model.models import _get_model
from src.evaluation.time_metrics import timeit
from src.utils.transforms import UNMIXING_TRANSFORMS
from torch.utils.data import DataLoader, TensorDataset
from src.utils.utils import get_central_pixel_spectrum
from src.evaluation.performance_metrics import UNMIXING_LOSSES, \
    UNMIXING_TRAIN_METRICS, calculate_unmixing_metrics


def fit(data: Dict[str, np.ndarray],
        model_name: str,
        dest_path: str,
        sample_size: int,
        n_classes: int,
        neighborhood_size: int,
        lr: float,
        batch_size: int,
        epochs: int,
        shuffle: bool,
        patience: int,
        endmembers_path: str,
        num_workers: int,
        seed: int):
    """
    Function for running experiments on various unmixing models,
    given a set of hyper parameters.

    :param data: The data dictionary containing
        the subsets for training and validation.
        First dimension of the datasets should be the number of samples.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param dest_path: Path to where all experiment runs will be saved as
        subdirectories in this given directory.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    :param neighborhood_size: Size of the spatial patch.
    :param lr: Learning rate for the model, i.e., regulates
        the size of the step in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param shuffle: Boolean indicating whether to shuffle datasets.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    :param endmembers_path: Path to the endmembers matrix file,
        containing the average reflectances for each endmember,
        i.e., the pure spectra.
    :param num_workers:
    :param seed: Seed for training reproducibility.
    """
    # Reproducibility:
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare data
    train_dict = data[enums.Dataset.TRAIN]
    test_dict = data[enums.Dataset.TEST]
    val_dict = data[enums.Dataset.VAL]

    min_, max_ = data[enums.DataStats.MIN], data[enums.DataStats.MAX]

    transformations = [transforms.MinMaxNormalize(min_=min_, max_=max_)]
    transformations += [t(**{'neighborhood_size': neighborhood_size}) for t
                        in UNMIXING_TRANSFORMS[model_name]]

    train_dict = transforms.apply_transformations(train_dict, transformations)
    val_dict = transforms.apply_transformations(val_dict, transformations)
    test_dict_transformed = transforms.apply_transformations(test_dict.copy(), transformations)

    train_data = DataLoader(TensorDataset(torch.from_numpy(train_dict[enums.Dataset.DATA]),
                                          torch.from_numpy(train_dict[enums.Dataset.LABELS])),
                            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_data = DataLoader(TensorDataset(torch.from_numpy(val_dict[enums.Dataset.DATA]),
                                        torch.from_numpy(val_dict[enums.Dataset.LABELS])),
                          batch_size=batch_size, num_workers=2)

    test_data = DataLoader(TensorDataset(torch.from_numpy(test_dict_transformed[enums.Dataset.DATA])),
                           batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_label = torch.from_numpy(test_dict[enums.Dataset.LABELS])

    input_shape = train_dict[enums.Dataset.DATA].shape
    print(input_shape)

    # get model
    model = _get_model(
        model_key=model_name,
        **{'in_features': 160,
           'n_class': n_classes,
           'input_size': sample_size,
           'endmembers': np.load(endmembers_path) if endmembers_path is not None else None})

    print(model_name)
    model = torchkeras.Model(model)
    model.summary(input_shape=input_shape[1:])
    model.compile(optimizer=torch.optim.Adam(model.parameters(), lr=lr),
                  loss_func=UNMIXING_LOSSES[model_name],
                  metrics_dict=UNMIXING_TRAIN_METRICS[model_name], device=device)
    # train
    history = model.fit(train_data=train_data,
                        val_data=val_data,
                        epochs=epochs,
                        patience=patience,
                        monitor="val_loss",
                        verbose=False,
                        save_path=os.path.join(dest_path, "save_model.pkl"))

    # evaluate
    if 'dcae' in model_name:
        model = _get_model(
            model_key=model_name+'_load',
            **{'in_features': 160,
               'n_class': n_classes,
               'input_size': sample_size,
               'endmembers': np.load(endmembers_path) if endmembers_path is not None else None})

        state_dict = torch.load(os.path.join(dest_path, "save_model.pkl"))
        state_dict.pop('net.fc3.weight')
        state_dict.pop('net.fc3.bias')

        model.load_state_dict(state_dict, strict=False)
        model = torchkeras.Model(model)
    else:
        model.load_state_dict(os.path.join(dest_path, "save_model.pkl"))

    model.compile(optimizer=torch.optim.Adam(model.parameters(), lr=lr),
                  loss_func=UNMIXING_LOSSES[model_name],
                  metrics_dict=UNMIXING_TRAIN_METRICS[model_name], device=device)

    predict = timeit(model.predict)
    y_pred, inference_time = predict(test_data)

    model_metrics = calculate_unmixing_metrics(**{
        'endmembers': torch.from_numpy(np.load(endmembers_path)).t().to(device)
        if endmembers_path is not None else None,
        'y_pred': y_pred.to(device),
        'y_true': test_label.to(device),
        'x_true': torch.from_numpy(get_central_pixel_spectrum(
            test_dict_transformed[enums.Dataset.DATA],
            neighborhood_size)).to(device)
    })
    model_metrics['inference_time'] = [inference_time]

    # save metrics
    io.save_metrics(dest_path=dest_path, filename='training_metrics.csv', metrics=history)
    io.save_metrics(dest_path=dest_path,
                    filename=enums.Experiment.INFERENCE_METRICS,
                    metrics=pd.DataFrame(model_metrics))