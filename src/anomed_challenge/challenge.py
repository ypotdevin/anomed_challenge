"""This module provides means to create challenges for the AnoMed competition
platform, focusing on the data and utility definition, avoiding any web
communication issues."""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, TypeAlias

import anomed_utils as utils
import numpy as np
import pandas as pd

__all__ = [
    "AnonymizationScheme",
    "discard_targets",
    "evaluate_membership_inference_attack",
    "evaluate_MIA",
    "InMemoryNumpyArrays",
    "NpzFromDisk",
    "NumpyDataset",
    "strict_binary_accuracy",
    "SupervisedLearningMIAChallenge",
    "TabularDataReconstructionChallenge",
]

_logger = logging.getLogger(__name__)


class NumpyDataset(ABC):
    """An abstract class defining a basic interface for handling datasets, based
    on Numpy arrays.

    It is intended for the common case of supervised learning, where you have
    feature arrays and target arrays. Inheriting classes only have to provide a
    `get` method. Default implementations for `__eq__`, `__repr__` and `__str__`
    are given."""

    @abstractmethod
    def get(self) -> tuple[np.ndarray, np.ndarray]:
        """An accessor to the feature array and the target array.

        Returns
        -------
        (X, y): tuple[np.ndarray, np.ndarray]
            X is the feature array, y is the target array.
        """
        pass

    def __eq__(self, other) -> bool:
        if isinstance(other, NumpyDataset):
            (X_self, y_self) = self.get()
            (X_other, y_other) = other.get()
            return np.array_equal(X_self, X_other) and np.array_equal(y_self, y_other)
        else:
            return False

    def __repr__(self) -> str:
        (X, y) = self.get()
        return f"NumpyDataset(X={repr(X)}, y={repr(y)})"

    def __str__(self) -> str:
        (X, y) = self.get()
        return f"NumpyDataset(X={str(X)}, y={str(y)})"

    @property
    def shapes(self) -> list[tuple[int, ...]]:
        (X, y) = self.get()
        return [X.shape, y.shape]

    @property
    def dtypes(self) -> list[np.dtype]:
        (X, y) = self.get()
        return [X.dtype, y.dtype]


class NpzFromDisk(NumpyDataset):
    """An instance of a NumpyDataset, suitable for .npz files saved to disk."""

    def __init__(
        self, npz_filepath: str | Path, X_label: str = "X", y_label: str = "y"
    ):
        """
        Parameters
        ----------
        npz_filepath : str | Path
            The path to the .npz file.
        X_label : str, optional
            The label of the array within the .npz file which should be treated
            as `X`, the feature array. By default "X".
        y_label : str, optional
            The label of the array within the .npz file which should be treated
            as `y`, the target array. By default "y".

        Notes
        -----
        Other arrays that might reside in the .npz file are ignored.
        """

        self._npz_filepath = Path(npz_filepath)
        self._X_label = X_label
        self._y_label = y_label

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self._npz_filepath) as data:
            X = data[self._X_label]
            y = data[self._y_label]
            return (X, y)


class InMemoryNumpyArrays(NumpyDataset):
    """An instance of a NumpyDataset for Numpy arrays that resides in the main
    memory."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            The feature array.
        y : np.ndarray
            The target array.
        """
        self._X = X
        self._y = y

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._X, self._y)


class SupervisedLearningMIAChallenge:
    """This class represents supervised learning ML challenges, where the threat
    model in concerned with membership inference attacks (MIA).

    Instances bundle training, tuning and validation datasets for anonymizers
    (privacy preserving ML models); member, non-member and evaluation datasets
    for deanonymizers (attacks on anonymizers); and also means of evaluating
    anonymizers  and deanonymizers. If you are able to define your challenge
    obeying this interface, you will be able to transform it into a AnoMed
    compatible web server "for free", by using
    `anomed_challenge.supervised_learning_MIA_challenge_server_factory`."""

    def __init__(
        self,
        training_data: NumpyDataset,
        tuning_data: NumpyDataset,
        validation_data: NumpyDataset,
        anonymizer_evaluator: Callable[[np.ndarray, np.ndarray], dict[str, float]],
        MIA_evaluator: Callable[[np.ndarray, np.ndarray], dict[str, float]],
        MIA_evaluation_dataset_length: int,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        training_data : NumpyDataset
            The dataset offered to peers, whose goal is to train anonymizers.
        tuning_data : NumpyDataset
            The dataset offered to peers, whose goals is to tune their
            anonymizers which are currently in training (this is also called
            "validation data" outside of the medical ML community).
        validation_data : NumpyDataset
            The dataset used to validate (in the regulatory sense) the
            performance of a fully trained anonymizer. Not to be confused with
            what is called "validation data" outside of the medical ML field –
            for that we use the term "tuning data".
        anonymizer_evaluator : Callable[[np.ndarray, np.ndarray], dict[str, float]]
            A way to evaluate the prediction (first argument) of an anonymizer
            compared to the ground truth (second argument, will be substituted
            by the target array (`y`) of `validation_data`).
        MIA_evaluator : Callable[[np.ndarray, np.ndarray], dict[str, float]]
            A way to evaluate the prediction of a membership inference attack
            (first argument) compared to the ground truth memberships (second
            argument, defined by the second component of
            `MIA_evaluation_data(...)`).
        MIA_evaluation_dataset_length : int, optional
            The number of members and also the number of non-members to include
            in the return value of `MIA_evaluation_data`. That means in total,
            the length of that dataset will be
            `2 * MIA_evaluation_dataset_length`. We suggest to set this to at
            least 100.
        seed : int | None, optional
            If given, use this seed to create `members` and `non_members`. By
            default `None`, which means obtain randomness non-deterministically
            at runtime.
        """
        self.training_data = training_data
        self.tuning_data = tuning_data
        self.validation_data = validation_data
        self._anonymizer_evaluator = anonymizer_evaluator
        self._MIA_evaluator = MIA_evaluator
        self._members_train: NumpyDataset = None  # type: ignore
        self._members_val: NumpyDataset = None  # type: ignore
        self._non_members_train: NumpyDataset = None  # type: ignore
        self._non_members_val: NumpyDataset = None  # type: ignore
        self.MIA_evaluation_dataset_length = MIA_evaluation_dataset_length
        if seed is None:
            self._seed = np.random.default_rng().integers(low=0, high=2**30)
        else:
            self._seed = seed

    @property
    def members(self) -> NumpyDataset:
        """A dataset to train membership inference attacks. It consists of
        features and corresponding targets that the model under attack has seen
        during training (therefore "members"), i.e. the data is a subset of
        `training_data`."""
        if self._members_train is None:
            self._init_members()
        return self._members_train

    def _init_members(self) -> None:
        (members1, members2) = _random_partition(self.training_data, seed=self._seed)
        self._members_train = members1
        self._members_val = members2
        _log_numpy_dataset_info(
            headline="Initialized members:",
            numpy_datasets=dict(
                members_train=self._members_train, members_val=self._members_val
            ),
        )

    @property
    def non_members(self) -> NumpyDataset:
        """A dataset to train membership inference attacks. It consists of
        features and corresponding targets that the model under attack has *not*
        seen during training (therefore "non_members"), i.e. the data is a
        subset of `validation_data`."""
        if self._non_members_train is None:
            self._init_non_members()
        return self._non_members_train

    def _init_non_members(self) -> None:
        (non_members1, non_members2) = _random_partition(
            self.validation_data,
            seed=self._seed,
        )
        self._non_members_train = non_members1
        self._non_members_val = non_members2
        _log_numpy_dataset_info(
            headline="Initialized non-members:",
            numpy_datasets=dict(
                non_members_train=self._non_members_train,
                non_members_val=self._non_members_val,
            ),
        )

    def MIA_evaluation_data(
        self,
        anonymizer: str,
        deanonymizer: str,
        data_split: Literal["tuning", "validation"],
    ) -> tuple[NumpyDataset, np.ndarray]:
        """A dataset and corresponding memberships to evaluate the success of a
        membership inference attack.

        The dataset is individual for the specific combination of `anonymizer`,
        `deanonymizer` and the setting of `data_split` (at least with high
        probability). Its size is determined by
        `2 * MIA_evaluation_dataset_length` (provided at initialization).

        Parameters
        ----------
        anonymizer : str
            The identifier of the anonymizer being under attack.
        deanonymizer : str
            The identifier of the membership inference attack.
        data_split : Literal["tuning", "validation"]
            Which ground truth to use for reference. There is one dataset
            determined to be used once for validation and another, disjoint
            dataset determined to be used multiple times for tuning.


        Returns
        -------
        (dataset, memberships) : tuple[NumpyDataset, np.ndarray]
            A dataset of the same width and dtype as the training, tuning and
            validation data (although not of the same length/height) and a
            boolean array with corresponding memberships (ground truth).
            If `memberships[i]` is `True`, `dataset[i] == (X[i], y[i])` is a
            member, i.e. part of the training dataset. If `memberships[i]` is
            `False`, it is not a member, i.e. part of the validation dataset.
        """
        if self._members_val is None:
            self._init_members()
        if self._non_members_val is None:
            self._init_non_members()
        return _create_membership_inference_evaluation_data(
            members=self._members_val,
            non_members=self._non_members_val,
            desired_length=self.MIA_evaluation_dataset_length,
            deanonymizer=deanonymizer,
            anonymizer=anonymizer,
            data_split=data_split,
        )

    def evaluate_anonymizer(
        self, prediction: np.ndarray, ground_truth: np.ndarray
    ) -> dict[str, float]:
        """Evaluate an anonymizer, governed by `anonymizer_evaluator` provided
        at initialization.

        Parameters
        ----------
        prediction : np.ndarray
            The prediction of the anonymizer being under evaluation, i.e. an
            array of the same dtype and shape as the target array of the
            validation dataset.
        ground_truth : np.ndarray
            The corresponding ground truth (target array of the validation
            dataset).

        Returns
        -------
        dict[str, float]
            A dictionary of evaluation metrics, depending on the return value of
            `anonymizer_evaluator`.
        """
        return self._anonymizer_evaluator(prediction, ground_truth)

    def evaluate_membership_inference_attack(
        self, prediction: np.ndarray, ground_truth: np.ndarray
    ) -> dict[str, float]:
        """Evaluate a membership inference attack, governed by `MIA_evaluator`
        provided at initialization.

        Parameters
        ----------
        prediction : np.ndarray
            The prediction of the membership inference attack being under
            evaluation, i.e. a boolean array of size
            `2 * MIA_evaluation_dataset_length`.
        ground_truth : np.ndarray
            The corresponding ground truth (second component of
            `MIA_evaluation_data(...)`).

        Returns
        -------
        dict[str, float]
            A dictionary of evaluation metrics, depending on the return value of
            `MIA_evaluator`.
        """
        return self._MIA_evaluator(prediction, ground_truth)


def _log_numpy_dataset_info(
    headline: str | None,
    numpy_datasets: dict[str, NumpyDataset],
    level: int = logging.DEBUG,
    logger: logging.Logger | None = None,
):
    if logger is None:
        logger = _logger
    indentation_spaces = 0
    if headline is not None:
        indentation_spaces = 4
        logger.log(level=level, msg=headline)
    for name, npds in numpy_datasets.items():
        message = (
            indentation_spaces * " "
        ) + f"{name}: shapes={npds.shapes}, dtypes={npds.dtypes}"
        logger.log(level=level, msg=message)


def _create_membership_inference_evaluation_data(
    members: NumpyDataset,
    non_members: NumpyDataset,
    desired_length: int,
    deanonymizer: str,
    anonymizer: str,
    data_split: Literal["tuning", "validation"],
) -> tuple[NumpyDataset, np.ndarray]:
    hash_data = anonymizer + deanonymizer
    _logger.debug(f"Using string `{hash_data}` to create seed deterministically.")
    seed = int(hashlib.sha256(hash_data.encode("utf-8")).hexdigest(), 16)

    # Split members and non-members into even parts (one for tuning, one for
    # validation)
    (members_subset_tuning, members_subset_validation) = _random_partition(
        members, seed=seed
    )
    (non_members_subset_tuning, non_members_subset_validation) = _random_partition(
        non_members, seed=seed
    )
    _log_numpy_dataset_info(
        headline="Shapes and dtypes of the even parts:",
        numpy_datasets=dict(
            members_subset_tuning=members_subset_tuning,
            members_subset_validation=members_subset_validation,
            non_members_subset_tuning=non_members_subset_tuning,
            non_members_subset_validation=non_members_subset_validation,
        ),
        level=logging.DEBUG,
    )

    # Pick the relevant part
    if data_split == "tuning":
        members_subset = members_subset_tuning
        non_members_subset = non_members_subset_tuning
    else:
        # data_split == "validation"
        members_subset = members_subset_validation
        non_members_subset = non_members_subset_validation
    _logger.debug(f"Selected datasets for {data_split}.")

    # Reduce the datasets to the desired size
    try:
        (members_subset, _) = _random_partition(
            members_subset, first_split_length=desired_length, seed=seed
        )
        (non_members_subset, _) = _random_partition(
            non_members_subset, first_split_length=desired_length, seed=seed
        )
    except ValueError:
        raise ValueError(
            "The members and/or non_members datasets do not contain enough "
            "elements to draw two disjoint MIA evaluation datasets (each of "
            f"length {2 * desired_length}) from them."
        )
    _log_numpy_dataset_info(
        headline="Shapes and dtypes of the reduced parts:",
        numpy_datasets=dict(
            members_subset=members_subset,
            non_members_subset=non_members_subset,
        ),
        level=logging.DEBUG,
    )

    # Go back to Numpy array representation for further processing
    (X_members, y_members) = members_subset.get()
    (X_non_members, y_non_members) = non_members_subset.get()
    members_mask = np.ones(desired_length, dtype=bool)
    nonmembers_mask = np.zeros(desired_length, dtype=bool)

    X = np.concatenate((X_members, X_non_members))
    y = np.concatenate((y_members, y_non_members))
    memberships = np.concatenate((members_mask, nonmembers_mask))

    [X, y, memberships] = utils.shuffles([X, y, memberships], seed=seed)
    dataset = InMemoryNumpyArrays(X=X, y=y)
    _log_numpy_dataset_info(
        logger=_logger,
        headline="Shapes and dtypes of the final evaluation parts:",
        numpy_datasets=dict(dataset=dataset),
        level=logging.DEBUG,
    )
    _logger.debug(
        f"    For memberships, the shape is {memberships.shape} and the dtype is "
        f"{memberships.dtype}."
    )

    return (dataset, memberships)


def _random_partition(
    data: NumpyDataset, first_split_length: float | int = 0.5, seed: int | None = None
) -> tuple[InMemoryNumpyArrays, InMemoryNumpyArrays]:
    """Lift `anomed_utils.random_partitions` to `NumpyDataset`.

    Parameters
    ----------
    data : NumpyDataset
        The dataset to partition.
    first_split_length : float | int, optional
        If `float`, this is the fraction the first dataset after partitioning
        should contain. The complement will be the second dataset. If `int`, the
        first dataset will contain exactly that number of elements. By default
        a float value of `0.5` (half the size).
    seed : int | None, optional
        The randomness to use for partitioning. By default `None`, which means
        obtain randomness non-deterministically at runtime.

    Returns
    -------
    (first_split, second_split) : tuple[InMemoryNumpyArrays, InMemoryNumpyArrays]
        The dataset splits that make up a partition. The length of `first_split`
        is directly determined by `first_split_length`, whereas the length of
        `second_split` is determined indirectly.

    Raises
    ------
    ValueError
        If one of these conditions is met:
        * The lengths of the feature array and the target array in `dataset` do
          not match.
        * The parameter `first_split_length` is a float but not between 0.0 and
          1.0.
        * The parameter `first_split_length` is an int but not between 0 and the
          dataset length itself.
        * The parameter is not even a float or an int.
    """
    (X, y) = data.get()
    n1, n2 = len(X), len(y)
    if n1 != n2:
        raise ValueError(f"Lengths of X ({n1}) and y ({n2}) do not match!")
    if isinstance(first_split_length, float):
        if 0.0 <= first_split_length <= 1.0:
            desired_length = int(first_split_length * n1)
        else:
            raise ValueError(
                "Length of first split must lie in interval [0.0, 1.0] to make sense."
            )
    elif isinstance(first_split_length, int):
        if 0 <= first_split_length <= n1:
            desired_length = first_split_length
        else:
            raise ValueError(
                "Length of first split must be between 0 and the length of the "
                "dataset itself."
            )
    else:
        raise TypeError(f"Cannot interpret length parameter {first_split_length}.")
    [(X1, X2), (y1, y2)] = utils.random_partitions(
        arrays=[X, y], total_length=n1, desired_length=desired_length, seed=seed
    )
    return (InMemoryNumpyArrays(X1, y1), InMemoryNumpyArrays(X2, y2))


def discard_targets(data: NumpyDataset) -> InMemoryNumpyArrays:
    """A dataset transformer which discards the target array, replacing in by an
    empty one.

    Use this function to avoid leaking targets/labels.

    Parameters
    ----------
    data : NumpyDataset
        The dataset to discard the features of.

    Returns
    -------
    InMemoryNumpyArrays
        A new dataset, containing the old feature array and an empty target
        array.
    """
    (X, _) = data.get()
    return InMemoryNumpyArrays(X, np.array([]))


def strict_binary_accuracy(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    """Calculate the 'strict' binary accuracy of the prediction with respect to
    the ground truth.

    By strict accuracy, we mean the fraction of the number of times where

        `prediction[i] == ground_truth[i]`,

    for `0 <= i < len(prediction)` is `True`, divided by `len(prediction)`. Note
    this is not the same as `sum(prediction==ground_truth)`, which is more
    forgiving for higher dimensional arrays (for one-dimensional arrays, it is
    equivalent though).

    Parameters
    ----------
    prediction : np.ndarray
        An estimator's prediction. Should have same shape and dtype as
        `ground_truth` and should not be empty.
    ground_truth : np.ndarray
        The respective ground truth. Should have same shape and dtype as
        `prediction` and should not be empty.

    Returns
    -------
    dict[str, float]
        A dictionary with key `accuracy` and a strict accuracy value.

    Raises
    ------
    ValueError
        If `prediction`, or `ground_truth` is empty, if their shape or if their
        dtype does not match.
    """
    if len(prediction) == 0 or len(ground_truth) == 0:
        raise ValueError(
            "Accuracy is undefined for empty `prediction` or empty `ground_truth`."
        )
    if not prediction.dtype == ground_truth.dtype:
        raise ValueError(
            f"Dtype mismatch of prediction ({prediction.dtype}) and ground_truth "
            f"({ground_truth.dtype})."
        )
    if not prediction.shape == ground_truth.shape:
        raise ValueError(
            f"Shape mismatch of prediction {prediction.shape} and ground_truth "
            f"{ground_truth.shape}"
        )
    matches = 0
    for i in range(len(prediction)):
        if np.array_equal(prediction[i], ground_truth[i]):
            matches += 1
    accuracy = matches / len(prediction)
    return dict(accuracy=accuracy)


def evaluate_membership_inference_attack(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    """Evaluate the prediction of a membership estimator in terms of binary
    accuracy, true positive rate and false positive rate.

    The ground truth and the corresponding dataset is provided by
    `SupervisedLearningMIAChallenge.MIA_evaluation_data` (dataset is first
    component, membership mask is second component).

    Parameters
    ----------
    prediction : np.ndarray
        A non-empty, one-dimensional boolean array containing the estimator's
        membership prediction.
    ground_truth : np.ndarray
        A non-empty, one-dimensional boolean array containing the true
        memberships.

    Returns
    -------
    dict[str, float]
        A dictionary of metrics, namely accuracy ('acc'), false positive rate
        ('fpr') and true positive rate ('tpr').

    Raises
    ------
    ValueError
        If `prediction` or `ground_truth` are empty arrays, or if they are not
        of the same length or not boolean arrays.
    """
    try:
        cm = utils.binary_confusion_matrix(prediction, ground_truth)
        tp = cm["tp"]
        n = len(prediction)
        acc = (tp + cm["tn"]) / n
        tpr = tp / (tp + cm["fn"])
        fpr = cm["fp"] / n
        return dict(acc=acc, tpr=tpr, fpr=fpr)
    except ZeroDivisionError:
        raise ValueError(
            "Can't evaluate MIA if `prediction` or `ground_truth` is empty."
        )


def evaluate_MIA(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """A shorter alias for `evaluate_membership_inference_attack`."""
    return evaluate_membership_inference_attack(
        prediction=prediction, ground_truth=ground_truth
    )


AnonymizationScheme: TypeAlias = Literal["same", "generalization", "microaggregation"]
"""What scheme does the anonymized data follow?

- same: Leaky data and anonymized data match regarding shape (columns) and
  datatype.
- generalization: Cardinal data will be represented by (min, max) ranges,
  nominal data will be one-hot encoded.
- microaggregation: TODO
"""


class TabularDataReconstructionChallenge:
    """This class represents challenges that aim to protect tabular data via
    anonymization, while maintaining certain utility. The threat model involves
    attackers that aim to reconstruct the tabular data from the anonymized data
    plus background knowledge."""

    def __init__(
        self,
        leaky_data: pd.DataFrame,
        background_knowledge: pd.DataFrame,
        utility_evaluator: Callable[
            [pd.DataFrame, AnonymizationScheme, pd.DataFrame], dict[str, float]
        ],
        privacy_evaluator: Callable[[pd.DataFrame, pd.DataFrame], dict[str, float]],
    ):
        """
        Parameters
        ----------
        leaky_data : pd.Dataframe
            The original tabular data which should be protected by
            anonymization.
        background_knowledge : pd.DataFrame
            Background knowledge that is accessible by attackers.
        utility_evaluator : Callable[[pd.DataFrame, AnonymizationScheme, DataFrame], dict[str, float]]
            How to evaluate the utility of anonymized data, depending of the
            kind of anonymization. The first argument is the anonymized data,
            the second argument is the anonymization scheme and the third
            argument is the original/leaky data for comparison.

            This function parameter is used, when `evaluate_utility` is called.
        privacy_evaluator : Callable[[pd.DataFrame, pd.DataFrame], dict[str, float]]
            How to evaluate the quality of reconstructed data, i.e. the success
            of the attacker. This is also an indirect measure of how privacy
            preserving the anonymized data (as an attack target) has been.
            The first argument is the reconstructed data, the second argument
            the leaky data, which the attack target's anonymized data depends
            on.

            This function parameter is used, when `evaluate_privacy` is called.
        """
        self.leaky_data = leaky_data
        self.background_knowledge = background_knowledge
        self._evaluate_utility = utility_evaluator
        self._evaluate_privacy = privacy_evaluator

    def evaluate_utility(
        self,
        anonymized_data: pd.DataFrame,
        anonymization_scheme: AnonymizationScheme,
        leaky_data: pd.DataFrame,
    ) -> dict[str, float]:
        return self._evaluate_utility(anonymized_data, anonymization_scheme, leaky_data)

    def evaluate_privacy(
        self,
        reconstructed_data: pd.DataFrame,
        leaky_data: pd.DataFrame,
    ) -> dict[str, float]:
        return self._evaluate_privacy(reconstructed_data, leaky_data)
