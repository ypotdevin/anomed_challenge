import tempfile

import numpy as np
import pytest

import anomed_challenge as challenge
from anomed_challenge import InMemoryNumpyArrays, NpzFromDisk, NumpyDataset
from anomed_challenge.challenge import (
    _random_partition,
)


@pytest.fixture()
def empty_ndarray() -> np.ndarray:
    return np.array([])


@pytest.fixture()
def three_elem_ndarray() -> np.ndarray:
    return np.arange(3)


@pytest.fixture()
def ten_elem_ndarray() -> np.ndarray:
    return np.arange(10)


@pytest.fixture()
def boolean_ndarray() -> np.ndarray:
    return np.array(5 * [True, False])


@pytest.fixture()
def two_dim_ndarray() -> np.ndarray:
    arr = np.random.default_rng(42).integers(low=0, high=10, size=(10, 10))
    return arr


@pytest.fixture()
def example_dataset() -> NumpyDataset:
    return InMemoryNumpyArrays(np.arange(10), np.arange(10))


@pytest.fixture()
def example_training_dataset() -> NumpyDataset:
    return InMemoryNumpyArrays(np.arange(0, 70), np.arange(0, 70))


@pytest.fixture()
def example_tuning_dataset() -> NumpyDataset:
    return InMemoryNumpyArrays(np.arange(70, 100), np.arange(70, 100))


@pytest.fixture()
def example_validation_dataset() -> NumpyDataset:
    return InMemoryNumpyArrays(np.arange(100, 120), np.arange(100, 120))


@pytest.fixture()
def empty_challenge(empty_ndarray) -> challenge.SupervisedLearningMIAChallenge:
    chal = challenge.SupervisedLearningMIAChallenge(
        training_data=empty_ndarray,
        tuning_data=empty_ndarray,
        validation_data=empty_ndarray,
        anonymizer_evaluator=challenge.strict_binary_accuracy,
        MIA_evaluator=challenge.evaluate_MIA,
        MIA_evaluation_dataset_length=0,
        seed=0,
    )
    return chal


def test_NpzFromDisk(empty_ndarray, ten_elem_ndarray):
    with tempfile.NamedTemporaryFile(mode="w+b") as tmp_file:
        np.savez_compressed(tmp_file, X=empty_ndarray, y=empty_ndarray)
        dataset = NpzFromDisk(tmp_file.name)
        assert _compare_dataset_components(empty_ndarray, empty_ndarray, dataset)
    with tempfile.NamedTemporaryFile(mode="w+b") as tmp_file:
        np.savez_compressed(tmp_file, X=ten_elem_ndarray, y=ten_elem_ndarray)
        dataset = NpzFromDisk(tmp_file.name)
        assert _compare_dataset_components(ten_elem_ndarray, ten_elem_ndarray, dataset)


def _compare_dataset_components(
    X: np.ndarray, y: np.ndarray, dataset: NumpyDataset
) -> bool:
    (X_, y_) = dataset.get()
    return np.array_equal(X, X_) and np.array_equal(y, y_)


def test_InMemoryNumpyArrays(empty_ndarray, ten_elem_ndarray):
    dataset = InMemoryNumpyArrays(empty_ndarray, empty_ndarray)
    assert _compare_dataset_components(empty_ndarray, empty_ndarray, dataset)
    dataset = InMemoryNumpyArrays(ten_elem_ndarray, ten_elem_ndarray)
    assert _compare_dataset_components(ten_elem_ndarray, ten_elem_ndarray, dataset)


def test_NumpyDataset_eq(example_training_dataset, example_validation_dataset):
    assert example_training_dataset == example_training_dataset
    assert example_training_dataset != example_validation_dataset
    assert example_training_dataset != 42


def test_NumpyDataset_repr(three_elem_ndarray):
    ds = InMemoryNumpyArrays(three_elem_ndarray, three_elem_ndarray)
    assert repr(ds) == "NumpyDataset(X=array([0, 1, 2]), y=array([0, 1, 2]))"
    assert str(ds) == "NumpyDataset(X=[0 1 2], y=[0 1 2])"


def test_NumpyDataset_shapes():
    ds = InMemoryNumpyArrays(np.arange(10).reshape(2, 5), np.arange(9).reshape(3, 3))
    assert ds.shapes == [(2, 5), (3, 3)]


def test_NumpyDataset_dtypes(three_elem_ndarray):
    ds = InMemoryNumpyArrays(
        np.arange(10, dtype=np.int_), np.arange(5, dtype=np.float_)
    )
    assert ds.dtypes == [np.int_, np.float_]


def test_discard_targets(example_dataset):
    discarded_dataset = challenge.discard_targets(example_dataset)
    (_, y) = discarded_dataset.get()
    assert len(y) == 0


def test_challenge_anonymizer_evaluation(
    empty_challenge, empty_ndarray, ten_elem_ndarray, two_dim_ndarray, boolean_ndarray
):
    with pytest.raises(ValueError):
        empty_challenge.evaluate_anonymizer(empty_ndarray, empty_ndarray)

    with pytest.raises(ValueError):
        empty_challenge.evaluate_anonymizer(ten_elem_ndarray, two_dim_ndarray)

    with pytest.raises(ValueError):
        empty_challenge.evaluate_anonymizer(boolean_ndarray, ten_elem_ndarray)

    altered_array = two_dim_ndarray.copy()
    for i in range(4):
        altered_array[i, i] = -1
    strict_acc = empty_challenge.evaluate_anonymizer(altered_array, two_dim_ndarray)
    assert strict_acc == dict(accuracy=0.6)


def test_challenge_MIA_evaluation(empty_challenge, empty_ndarray, boolean_ndarray):
    with pytest.raises(ValueError):
        empty_challenge.evaluate_membership_inference_attack(
            empty_ndarray.astype(bool), empty_ndarray.astype(bool)
        )

    with pytest.raises(ValueError):
        empty_challenge.evaluate_membership_inference_attack(
            boolean_ndarray, np.concatenate([boolean_ndarray, boolean_ndarray])
        )

    altered_array = boolean_ndarray.copy()
    for i in range(5):
        altered_array[i] = True
    MIA_evaluation = empty_challenge.evaluate_membership_inference_attack(
        altered_array, boolean_ndarray
    )
    assert MIA_evaluation == dict(acc=0.8, tpr=1.0, fpr=0.2)


def test_SupervisedLearningMIAChallenge(
    example_training_dataset, example_tuning_dataset, example_validation_dataset
):
    chal = challenge.SupervisedLearningMIAChallenge(
        training_data=example_training_dataset,
        tuning_data=example_tuning_dataset,
        validation_data=example_validation_dataset,
        anonymizer_evaluator=challenge.strict_binary_accuracy,
        MIA_evaluator=challenge.evaluate_MIA,
        MIA_evaluation_dataset_length=5,
    )
    assert _is_subset_dataset(
        chal.members, chal.training_data
    ), "Expected the `members` dataset to be derived from training data."
    assert _is_subset_dataset(
        chal.non_members, chal.validation_data
    ), "Expected the `non_members` dataset to be derived from validation data."
    (mia_eval_dataset_tuning, _) = chal.MIA_evaluation_data("a", "b", "tuning")
    (mia_eval_dataset_validation, _) = chal.MIA_evaluation_data("a", "b", "validation")
    assert _is_subset_dataset(
        mia_eval_dataset_tuning,
        _join_datasets(chal.training_data, chal.validation_data),
    ), "Expected the MIA evaluation data to be derived from both, training and "
    "validation data."
    assert _is_subset_dataset(
        mia_eval_dataset_validation,
        _join_datasets(chal.training_data, chal.validation_data),
    ), "Expected the MIA evaluation data to be derived from both, training and "
    "validation data."
    assert _are_disjoint_datasets(
        mia_eval_dataset_tuning, chal.members
    ) and _are_disjoint_datasets(
        mia_eval_dataset_tuning, chal.non_members
    ), "Expected the MIA evaluation data to be disjoint from both, the members and "
    "non_members."
    assert _are_disjoint_datasets(
        mia_eval_dataset_validation, chal.members
    ) and _are_disjoint_datasets(
        mia_eval_dataset_validation, chal.non_members
    ), "Expected the MIA evaluation data to be disjoint from both, the members and "
    "non_members."


def _join_datasets(left: NumpyDataset, right: NumpyDataset) -> InMemoryNumpyArrays:
    (X_left, y_left) = left.get()
    (X_right, y_right) = right.get()
    return InMemoryNumpyArrays(
        X=np.concatenate([X_left, X_right]), y=np.concatenate([y_left, y_right])
    )


def _are_disjoint_datasets(left: NumpyDataset, right: NumpyDataset) -> bool:
    (left_X, left_y) = left.get()
    (right_X, right_y) = right.get()
    return _are_disjoint(left_X, right_X) and _are_disjoint(left_y, right_y)


def _are_disjoint(left: np.ndarray, right: np.ndarray) -> bool:
    no_left_in_right = all([e not in right for e in left])
    no_right_in_left = all([e not in left for e in right])
    return no_left_in_right and no_right_in_left


def _is_subset_dataset(subset: NumpyDataset, superset: NumpyDataset) -> bool:
    (subset_X, subset_y) = subset.get()
    (superset_X, superset_y) = superset.get()
    return _is_subset(subset_X, superset_X) and _is_subset(subset_y, superset_y)


def _is_subset(subset: np.ndarray, superset: np.ndarray) -> bool:
    return all([e in superset for e in subset])


def test_failing_random_partition(
    example_dataset, three_elem_ndarray, ten_elem_ndarray
):
    data = InMemoryNumpyArrays(three_elem_ndarray, ten_elem_ndarray)
    with pytest.raises(ValueError):
        _random_partition(data)
    with pytest.raises(ValueError):
        _random_partition(example_dataset, first_split_length=-1.0)
    with pytest.raises(ValueError):
        _random_partition(example_dataset, first_split_length=11)
    with pytest.raises(TypeError):
        _random_partition(example_dataset, first_split_length="a")  # type: ignore


def test_failing_create_membership_inference_evaluation_data(
    example_training_dataset, example_tuning_dataset, example_validation_dataset
):
    chal = challenge.SupervisedLearningMIAChallenge(
        training_data=example_training_dataset,
        tuning_data=example_tuning_dataset,
        validation_data=example_validation_dataset,
        anonymizer_evaluator=challenge.strict_binary_accuracy,
        MIA_evaluator=challenge.evaluate_MIA,
        MIA_evaluation_dataset_length=500,
    )
    with pytest.raises(ValueError):
        chal.MIA_evaluation_data(
            anonymizer="example-anonymizer",
            deanonymizer="example-deanonymizer",
            data_split="tuning",
        )
