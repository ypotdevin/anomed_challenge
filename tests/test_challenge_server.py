import anomed_utils as utils
import falcon
import falcon.testing
import numpy as np
import pandas as pd
import pytest
import urllib3
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


@pytest.fixture()
def example_float_df():
    return pd.DataFrame(data=np.arange(10, dtype=float))


@pytest.fixture()
def example_int_df():
    return pd.DataFrame(data=np.arange(start=10, stop=20, dtype=int))


@pytest.fixture()
def example_tabl_data_reconstr_challenge(example_float_df, example_int_df):
    return challenge.TabularDataReconstructionChallenge(
        leaky_data=example_float_df,
        background_knowledge=example_int_df,
        utility_evaluator=lambda x, y, z: dict(truth=42),
        privacy_evaluator=lambda x, y: dict(half_truth=21),
    )


@pytest.fixture()
def data_reconstruction_client(example_tabl_data_reconstr_challenge):
    challenge_app = challenge.tabular_data_reconstruction_challenge_server_factory(
        challenge_obj=example_tabl_data_reconstr_challenge
    )
    return testing.TestClient(challenge_app)


def test_availability(client):
    assert _is_available(client)


def _is_available(client_) -> bool:
    message = {"message": "Challenge server is alive!"}
    response = client_.simulate_get("/")
    return response.json == message


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


def test_tabular_data_reconstruction_challenge_server_factory(
    data_reconstruction_client,
    example_tabl_data_reconstr_challenge,
    example_float_df,
    mocker,
):
    assert _is_available(data_reconstruction_client)
    assert _get_and_compare_df(
        data_reconstruction_client,
        "/data/anonymizer/leaky",
        example_tabl_data_reconstr_challenge.leaky_data,
    )
    assert _get_and_compare_df(
        data_reconstruction_client,
        "/data/deanonymizer/background-knowledge",
        example_tabl_data_reconstr_challenge.background_knowledge,
    )
    # TODO: Test evaluation (remember to mock submission requests)
    body, headers = _serialize_anonymized_data(
        anon_data=example_float_df, anon_scheme="same"
    )
    _check_utility(
        data_reconstruction_client,
        mocker,
        route="/utility/anonymizer",
        headers=headers,
        body=body,
        expected_status_code=201,
        expected_evaluation=dict(truth=42),
    )

    _check_utility(
        data_reconstruction_client,
        mocker,
        route="/utility/deanonymizer",
        headers=None,
        body=utils.dataframe_to_bytes(example_float_df),
        expected_status_code=201,
        expected_evaluation=dict(half_truth=21),
    )


def _serialize_anonymized_data(
    anon_data: pd.DataFrame, anon_scheme: challenge.AnonymizationScheme
) -> tuple[bytes, dict[str, str]]:
    fields = {
        "anon_data": utils.dataframe_to_bytes(anon_data),
        "anon_scheme": anon_scheme,
    }
    body, content_type_header = urllib3.encode_multipart_formdata(fields)
    headers = {
        "Content-Type": content_type_header,
    }
    return (body, headers)


def _check_utility(
    client_,
    mocker_,
    route: str,
    headers: dict[str, str] | None,
    body: bytes,
    expected_status_code: int,
    expected_evaluation: dict[str, float],
) -> None:
    mock_response = mocker_.MagicMock()
    mock_response.status_code = 201
    mock = mocker_.patch("requests.post", return_value=mock_response)

    response = client_.simulate_post(route, headers=headers, body=body)

    mock.assert_called_once()
    assert response.status_code == expected_status_code
    assert response.json == expected_evaluation


def _get_and_compare_df(
    client_: falcon.testing.TestClient, route: str, df: pd.DataFrame
) -> bool:
    result = client_.simulate_get(route)
    remote_df = _df_from_response(result)
    return df.equals(remote_df)


def _df_from_response(result: falcon.testing.Result) -> pd.DataFrame:
    if result.status_code == 200:
        df = utils.bytes_to_dataframe(result.content)
    else:
        raise ValueError("Faulty response.")
    return df
