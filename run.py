"""
Created on Jan 29, 2021

@file: runner.py
@desc: Run experiments given set of hyperparameters for the unmixing problem.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import os
from config.get_config import get_config

from src.model import enums
from src.utils.utils import parse_train_size, subsample_test_set
from src.utils import prepare_data, artifacts_reporter
from src.model import train_evaluate
from src.model.models import pixel_based_cnn, cube_based_cnn, pixel_based_dcae

# Literature hyperparameters settings:
NEIGHBORHOOD_SIZES = {
    # cube_based_dcae.__name__: 5,
    cube_based_cnn.__name__: 3,
    #
    # attention_cube_based_dcae.__name__: 5,
    # attention_cube_based_cnn.__name__: 3
}

LEARNING_RATES = {
    pixel_based_dcae.__name__: 0.001,
    # cube_based_dcae.__name__: 0.0005,

    pixel_based_cnn.__name__: 0.01,
    cube_based_cnn.__name__: 0.001
    #
    # attention_pixel_based_dcae.__name__: 0.001,
    # attention_cube_based_dcae.__name__: 0.0005,
    #
    # attention_pixel_based_cnn.__name__: 0.01,
    # attention_cube_based_cnn.__name__: 0.001
}


def run_experiments(*,
                    data_file_path: str,
                    ground_truth_path: str = None,
                    train_size: int or float,
                    val_size: float = 0.1,
                    sub_test_size: int = None,
                    channels_idx: int = -1,
                    neighborhood_size: int = None,
                    n_runs: int = 1,
                    model_name: str,
                    dest_path: str = None,
                    sample_size: int,
                    n_classes: int,
                    lr: float = None,
                    batch_size: int = 256,
                    epochs: int = 100,
                    shuffle: bool = True,
                    patience: int = 15,
                    endmembers_path: str = None):
    """
    Function for running experiments on unmixing given a set of hyper parameters.

    :param data_file_path: Path to the data file. Supported types are: .npy.
    :param ground_truth_path: Path to the ground-truth data file.
    :param train_size: If float, should be between 0.0 and 1.0.
        If int, specifies the number of samples in the training set.
        Defaults to 0.8
    :type train_size: Union[int, float]
    :param val_size: Should be between 0.0 and 1.0. Represents the
        percentage of samples from the training set to be
        extracted as a validation set.
        Defaults to 0.1.
    :param sub_test_size: Number of pixels to subsample the test set
        instead of performing the inference on all
        samples that are not in the training set.
    :param channels_idx: Index specifying the channels position in the provided data.
    :param neighborhood_size: Size of the spatial patch.
    :param n_runs: Number of total experiment runs.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param dest_path: Path to where all experiment runs will be saved as
        subdirectories in this directory.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
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
    """
    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(dest_path,
                                            '{}_{}'.format(enums.Experiment.EXPERIMENT, str(experiment_id)))
        os.makedirs(experiment_dest_path, exist_ok=True)

        # Apply default literature hyper parameters:
        if neighborhood_size is None and model_name in NEIGHBORHOOD_SIZES:
            neighborhood_size = NEIGHBORHOOD_SIZES[model_name]
        if lr is None and model_name in LEARNING_RATES:
            lr = LEARNING_RATES[model_name]

        data = prepare_data.main(data_file_path=data_file_path,
                                 ground_truth_path=ground_truth_path,
                                 train_size=parse_train_size(train_size),
                                 val_size=val_size,
                                 background_label=-1,
                                 channels_idx=channels_idx,
                                 neighborhood_size=neighborhood_size,
                                 seed=experiment_id)
        if sub_test_size is not None:
            subsample_test_set(data[enums.Dataset.TEST], sub_test_size)
        train_evaluate.fit(model_name=model_name,
                           dest_path=experiment_dest_path,
                           data=data,
                           neighborhood_size=neighborhood_size,
                           n_classes=n_classes,
                           sample_size=sample_size,
                           lr=lr,
                           batch_size=batch_size,
                           epochs=epochs,
                           shuffle=shuffle,
                           patience=patience,
                           endmembers_path=endmembers_path,
                           num_workers=2,
                           seed=experiment_id)
    #
    # artifacts_reporter.collect_artifacts_report(
    #     experiments_path=dest_path,
    #     dest_path=dest_path)


if __name__ == '__main__':
    args = get_config(filename='./config/config.json')

    for model_name in args.model_names:
        for i in range(len(args.dataset)):
            dest_path = os.path.join(args.save_path,
                                     '{}_{}'.format(str(model_name), str(args.dataset[i])))

            base_path = os.path.join(args.path, args.dataset[i])
            data_file_path = os.path.join(base_path, args.dataset[i] + '.npy')
            ground_truth_path = os.path.join(base_path, args.dataset[i] + '_gt.npy')

            if "cnn" in model_name:
                endmembers_path = None
            else:
                endmembers_path = os.path.join(base_path, args.dataset[i] + '_m.npy')

            if args.dataset[i] == 'urban':
                sample_size, n_classes = 162, 6
            else:
                sample_size, n_classes = 157, 4

            run_experiments(data_file_path=data_file_path,
                            ground_truth_path=ground_truth_path,
                            endmembers_path=endmembers_path,
                            dest_path=dest_path,
                            train_size=args.train_size[i],
                            sub_test_size=args.test_size[i],
                            val_size=args.val_size,
                            model_name=model_name,
                            sample_size=sample_size,
                            n_classes=n_classes,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            patience=args.patience,
                            n_runs=args.n_runs)
