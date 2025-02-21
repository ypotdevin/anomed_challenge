import anomed_utils as utils
import numpy as np
import pytest
from falcon import testing

import anomed_challenge as challenge


@pytest.fixture()
def example_training_dataset() -> challenge.NumpyDataset:
    return challenge.InMemoryNumpyArrays(np.arange(0, 70), np.arange(0, 70))


@pytest.fixture()
def example_tuning_dataset() -> challenge.NumpyDataset:
    return challenge.InMemoryNumpyArrays(np.arange(70, 100), np.arange(70, 100))


@pytest.fixture()
def example_validation_dataset() -> challenge.NumpyDataset:
    return challenge.InMemoryNumpyArrays(np.arange(100, 120), np.arange(100, 120))


@pytest.fixture()
def client(
    example_training_dataset, example_tuning_dataset, example_validation_dataset
):
    test_challenge = challenge.SupervisedLearningMIAChallenge(
        training_data=example_training_dataset,
        tuning_data=example_tuning_dataset,
        validation_data=example_validation_dataset,
        anonymizer_evaluator=challenge.strict_binary_accuracy,
        MIA_evaluator=challenge.evaluate_MIA,
        MIA_evaluation_dataset_length=5,
    )
    return testing.TestClient(
        challenge.supervised_learning_MIA_challenge_server_factory(
            challenge_obj=test_challenge
        )
    )


def test_availability(client):
    message = {"message": "Challenge server is alive!"}
    response = client.simulate_get("/")
    assert response.json == message


def _get_and_compare(client_, route: str, dataset: challenge.NumpyDataset) -> bool:
    response = client_.simulate_get(route)
    remote_dataset = _dataset_from_result(response)
    return dataset == remote_dataset


def _dataset_from_result(result) -> challenge.InMemoryNumpyArrays:
    if result.status_code == 200:
        arrays = utils.bytes_to_named_ndarrays(result.content)
    else:
        raise ValueError("Faulty response.")
    dataset = challenge.InMemoryNumpyArrays(**arrays)
    return dataset


def test_get_anon_training_data(client, example_training_dataset):
    assert _get_and_compare(
        client, "/data/anonymizer/training", example_training_dataset
    )


def test_get_anon_tuning_data(client, example_tuning_dataset):
    assert _get_and_compare(
        client,
        "/data/anonymizer/tuning",
        challenge.discard_targets(example_tuning_dataset),
    )


def test_get_anon_validation_data(client, example_validation_dataset):
    assert _get_and_compare(
        client,
        "/data/anonymizer/validation",
        challenge.discard_targets(example_validation_dataset),
    )


def test_get_deanon_members(client):
    assert _route_represents_dataset(client, "/data/deanonymizer/members")


def _route_represents_dataset(
    client_, route: str, query_params: dict[str, str] | None = None
) -> bool:
    result = client_.simulate_get(route, params=query_params)
    try:
        _ = _dataset_from_result(result)
        return result.status_code == 200
    except ValueError:
        return False


def test_get_deanon_non_members(client):
    assert _route_represents_dataset(client, "/data/deanonymizer/non-members")


def test_get_MIA_eval_data(client):
    assert _route_represents_dataset(
        client,
        "/data/attack-success-evaluation",
        query_params=dict(
            deanonymizer="example-deanonymizer",
            anonymizer="example-anonymizer",
            data_split="tuning",
        ),
    )


def test_successful_anonymizer_utility(client, mocker):
    prediction = dict(prediction=np.arange(100, 120))
    payload: bytes = utils.named_ndarrays_to_bytes(prediction)
    expected_evaluation = dict(accuracy=1.0)

    mock_response = mocker.MagicMock()
    mock_response.status_code = 201
    mock = mocker.patch("requests.post", return_value=mock_response)

    response = client.simulate_post(
        "/utility/anonymizer",
        params=dict(anonymizer="example-anonymizer", data_split="validation"),
        body=payload,
    )

    mock.assert_called_once()
    assert response.status_code == 201
    assert response.json == expected_evaluation


def test_failing_anonymizer_utility(client):
    response = client.simulate_post(
        "/utility/anonymizer",
        params=dict(anonymizer="example-anonymizer", data_split="validation"),
        body=utils.named_ndarrays_to_bytes(dict(not_prediction=np.arange(10))),
    )
    assert response.status_code == 400
    assert (
        response.json["description"] == "Expected a NumPy array with name 'prediction'."
    )
    response = client.simulate_post(
        "/utility/anonymizer",
        params=dict(anonymizer="example-anonymizer", data_split="validation"),
        body=b"fail",
    )
    assert response.status_code == 400
    assert (
        response.json["description"] == "Expected a NumPy array with name 'prediction'."
    )


def test_deanonymizer_utility(client, mocker):
    prediction = np.zeros(shape=(10,), dtype=np.bool_)
    payload: bytes = utils.named_ndarrays_to_bytes(dict(prediction=prediction))
    expected_evaluation = dict(acc=0.5, fpr=0.0, tpr=0.0)

    mock_response = mocker.MagicMock()
    mock_response.status_code = 201
    mock = mocker.patch("requests.post", return_value=mock_response)

    response = client.simulate_post(
        "/utility/deanonymizer",
        params=dict(
            anonymizer="example-anonymizer",
            deanonymizer="example-deanonymizer",
            data_split="validation",
        ),
        body=payload,
    )

    mock.assert_called_once()
    assert response.status_code == 201
    assert response.json == expected_evaluation
