"""This module provides means to create web applications that host challenges at
and integrates well with the AnoMed competition platform.

Your first goal should be to make use of a web app factory, e.g.
`supervised_learning_MIA_challenge_server_factory`, to avoid dealing with web
programming issues. If this does not suit your use case, have a look at the more
basic resource building block (e.g. `StaticNumpyDataResource` or
`UtilityResource`.)
"""

import json
import logging
import os
from typing import Any, Callable

import anomed_utils as utils
import falcon
import numpy as np
import pandas as pd
import requests

from . import challenge
from .challenge import NumpyDataset

__all__ = [
    "DataReconstructionPrivacyResource",
    "DataReconstructionUtilityResource",
    "DynamicNumpyDataResource",
    "StaticJSONResource",
    "StaticNumpyDataResource",
    "supervised_learning_MIA_challenge_server_factory",
    "tabular_data_reconstruction_challenge_server_factory",
    "UtilityResource",
]


class StaticJSONResource:
    """Any JSON serializable object, representing a "static" resource (i.e. a
    resource that does not depend on request parameters).

    The object will be represented as a plain JSON string, when a GET request is
    invoked."""

    def __init__(self, obj: Any):
        """
        Parameters
        ----------
        obj : Any
            A JSON serializable object, i.e. is should be compatible with
            `json.dumps`.
        """
        self._obj = obj

    def on_get(self, _, resp: falcon.Response):
        resp.text = json.dumps(self._obj)


class StaticNumpyDataResource:
    """A static NumPy dataset resource, responding to GET requests.

    The arrays (features and targets) will be serialized using
    `anomed_utils.named_ndarrays_to_bytes` and may be de-serialized using
    `anomed_utils.bytes_to_named_ndarrays`. The names of the arrays are 'X' for
    the features and 'y' for the targets."""

    def __init__(self, data: NumpyDataset) -> None:
        self._data = data

    def on_get(self, _, resp: falcon.Response) -> None:
        _add_ds_to_resp(self._data, resp)


def _add_ds_to_resp(ds: NumpyDataset, resp: falcon.Response) -> None:
    (X, y) = ds.get()
    arrays = dict(X=X, y=y)
    resp.data = utils.named_ndarrays_to_bytes(arrays)
    resp.content_type = "application/octet-stream"


class DynamicNumpyDataResource:
    """A dynamic NumPy dataset resource, i.e. the specific dataset content
    depends on GET request parameters.

    The arrays (features and targets) will be serialized using
    `anomed_utils.named_ndarrays_to_bytes` and may be de-serialized using
    `anomed_utils.bytes_to_named_ndarrays`. The names of the arrays are 'X' for
    the features and 'y' for the targets."""

    def __init__(
        self, individual_data_provider: Callable[[falcon.Request], NumpyDataset]
    ) -> None:
        """
        Parameters
        ----------
        individual_data_provider : Callable[[falcon.Request], NumpyDataset]
            A function that, based on the GET request, returns a specific
            `NumpyDataset`.
        """
        self._individual_data_provider = individual_data_provider

    def on_get(self, req, resp: falcon.Response) -> None:
        individualized_data = self._individual_data_provider(req)
        _add_ds_to_resp(individualized_data, resp)


class UtilityResource:
    """Evaluate the prediction of an estimator with respect to the ground truth
    and report the result statistics as a JSON string.

    When receiving a POST request, extract a prediction array from the request's
    body, compare it to the ground truth and report the evaluation results. The
    prediction array must be byte encoded compatible with
    `anomed_utils.bytes_to_named_ndarrays` and have the name 'prediction'. The
    ground truth may depend on request parameters. The evaluation metrics used,
    are defined at runtime.
    """

    def __init__(
        self,
        target_data_provider: Callable[[falcon.Request], np.ndarray],
        evaluator: Callable[[np.ndarray, np.ndarray], dict[str, float]],
        submitter: Callable[[falcon.Request, dict[str, float]], None],
    ) -> None:
        """
        Parameters
        ----------
        target_data_provider : Callable[[falcon.Request], np.ndarray]
            A callable returning a reference array (ground truth), depending on
            the request parameters. It is expected that this callable takes care
            of request (parameter) validation and raises a
            `falcon.HTTPBadRequest` in case there are
        evaluator : Callable[[np.ndarray, np.ndarray], dict[str, float]]
            A callable comparing the prediction (first argument, extracted from
            the POST request) with the ground truth (second argument, provided
            by `target_data_provider`), generating evaluation statistics (a
            dictionary of float values). That dictionary will be the JSON string
            response.
        submitter : Callable[falcon.Request, [dict[str, float]], None]
            A callable which will submit the statistics generated by `evaluator`
            to the AnoMed competition platform (where it will be persisted). It
            may obtain submission-critical information from the request. It is
            expected that the this callable raises an
            `falcon.HTTPInternalServerError`, if the submission fails.
        """
        self._evaluation_data_provider = target_data_provider
        self._evaluator = evaluator
        self._submitter = submitter

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        """
        Raises
        ------
        falcon.HTTPBadRequest
            If extracting an array with name 'prediction' from the POST request
            body fails.
        """
        try:
            array_payload = utils.bytes_to_named_ndarrays(req.bounded_stream.read())
            self._validate_array_payload(array_payload)
            prediction = array_payload["prediction"]
        except (KeyError, ValueError) as error:
            logging.error("Expected a NumPy array with name 'prediction'.")
            logging.error(error)
            raise falcon.HTTPBadRequest(
                title="Malformed request.",
                description="Expected a NumPy array with name 'prediction'.",
            )
        y = self._evaluation_data_provider(req)
        evaluation = self._evaluator(prediction, y)
        self._submitter(req, evaluation)
        _add_evaluation_to_resp(evaluation=evaluation, resp=resp)

    def _validate_array_payload(self, array_payload: dict[str, np.ndarray]) -> None:
        if "prediction" not in array_payload:
            raise KeyError("'prediction'")


def _add_evaluation_to_resp(
    evaluation: dict[str, float], resp: falcon.Response, status_code: int = 201
) -> None:
    resp.text = json.dumps(evaluation)
    resp.status_code = status_code


def supervised_learning_MIA_challenge_server_factory(
    challenge_obj: challenge.SupervisedLearningMIAChallenge,
) -> falcon.App:
    """A factory to create a web application object which hosts a
    `challenge.SupervisedLearningMIAChallenge`, currently the most basic use
    case of challenges for the AnoMed competition platform.

    By using this factory, you don't have to worry any web-programming issues,
    as they are hidden from you. The generated web app will feature the
    following routes:

    * [GET] `/`
    * [GET] `/data/anonymizer/training`
    * [GET] `/data/anonymizer/tuning`
    * [GET] `/data/anonymizer/validation`
    * [GET] `/data/deanonymizer/members`
    * [GET] `/data/deanonymizer/non-members`
    * [GET] `/data/attack-success-evaluation`
    * [POST] `/utility/anonymizer`
    * [POST] `/utility/deanonymizer`

    Parameters
    ----------
    challenge_obj : challenge.SupervisedLearningMIAChallenge
        A supervised learning, MIA threat model challenge object providing the
        necessary data and means of utility and privacy evaluation.

    Returns
    -------
    falcon.App
        A web application object based on the falcon web framework.
    """
    app = falcon.App()
    app.add_route("/", StaticJSONResource(dict(message="Challenge server is alive!")))
    app.add_route(
        "/data/anonymizer/training",
        StaticNumpyDataResource(challenge_obj.training_data),
    )
    app.add_route(
        "/data/anonymizer/tuning",
        StaticNumpyDataResource(challenge.discard_targets(challenge_obj.tuning_data)),
    )
    app.add_route(
        "/data/anonymizer/validation",
        StaticNumpyDataResource(
            challenge.discard_targets(challenge_obj.validation_data)
        ),
    )
    app.add_route(
        "/data/deanonymizer/members", StaticNumpyDataResource(challenge_obj.members)
    )

    app.add_route(
        r"/data/deanonymizer/non-members",
        StaticNumpyDataResource(challenge_obj.non_members),
    )
    app.add_route(
        r"/data/attack-success-evaluation",
        DynamicNumpyDataResource(
            lambda req: challenge_obj.MIA_evaluation_data(
                anonymizer=req.get_param("anonymizer", required=True),
                deanonymizer=req.get_param("deanonymizer", required=True),
                data_split=req.get_param("data_split", required=True),  # type: ignore
            )[0]
        ),
    )
    app.add_route(
        "/utility/anonymizer",
        UtilityResource(
            target_data_provider=lambda req: challenge_obj.tuning_data.get()[1]
            if req.get_param("data_split", required=True) == "tuning"
            else challenge_obj.validation_data.get()[1],
            evaluator=challenge_obj.evaluate_anonymizer,
            # TODO: Change this to a serious submitter
            submitter=_demo_anonymizer_submitter,
        ),
    )
    app.add_route(
        "/utility/deanonymizer",
        UtilityResource(
            target_data_provider=lambda req: challenge_obj.MIA_evaluation_data(
                anonymizer=req.get_param("anonymizer", required=True),
                deanonymizer=req.get_param("deanonymizer", required=True),
                data_split=req.get_param("data_split", required=True),  # type: ignore
            )[1],
            evaluator=challenge_obj.evaluate_membership_inference_attack,
            # TODO: Change this to a serious submitter
            submitter=_demo_deanonymizer_submitter,
        ),
    )
    return app


def _demo_anonymizer_submitter(req, evaluation) -> None:
    anomed_hostname = os.getenv("ANOMED_HOST")
    url = f"http://{anomed_hostname}/submissions/anonymizer-evaluation-results"
    demo_evaluation = {
        "secret": "",
        "anonymizer": "example-anonymizer",
        "mae": 0.5,
        "rmse": 0.5,
        "coeff_determ": 0.5,
        "accuracy": 0.5,
        "auc": 0.5,
    }
    requests.post(url=url, json=demo_evaluation)


def _demo_deanonymizer_submitter(req, evaluation) -> None:
    anomed_hostname = os.getenv("ANOMED_HOST")
    url = f"http://{anomed_hostname}/submissions/deanonymizer-evaluation-results"
    demo_evaluation = {
        "secret": "",
        "deanonymizer": "example-deanonymizer",
        "fpr": 0.5,
        "tpr": 0.5,
    }
    requests.post(url=url, json=demo_evaluation)


class StaticDataFrameResource:
    """A static pandas `DataFrame` resource, responding to GET requests.

    The `DataFrame` (features and targets) will be serialized using
    `anomed_utils.named_ndarrays_to_bytes` and may be de-serialized using
    `anomed_utils.bytes_to_named_ndarrays`. The names of the arrays are 'X' for
    the features and 'y' for the targets."""

    def __init__(
        self,
        df: pd.DataFrame,
    ) -> None:
        self._df = df

    def on_get(self, req: falcon.Request, resp: falcon.Response) -> None:
        _add_df_to_resp(self._df, resp)


def _add_df_to_resp(df: pd.DataFrame, resp: falcon.Response) -> None:
    resp.data = utils.dataframe_to_bytes(df)
    resp.content_type = "application/octet-stream"


class DataReconstructionUtilityResource:
    def __init__(
        self,
        challenge_obj: challenge.TabularDataReconstructionChallenge,
        evaluation_submitter: Callable[[falcon.Request, dict[str, float]], None],
    ) -> None:
        self._challenge_obj = challenge_obj
        self._submit_evaluation = evaluation_submitter

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        # read anon_data and anon_format from request, evaluate their utility
        form = req.get_media()
        anon_data = anon_scheme = None
        for part in form:
            if part.name == "anon_data":
                anon_data = utils.bytes_to_dataframe(part.data)
            if part.name == "anon_scheme":
                anon_scheme = part.text
        utility = self._challenge_obj.evaluate_utility(
            anonymized_data=anon_data,  # type: ignore
            anonymization_scheme=anon_scheme,  # type: ignore
            leaky_data=self._challenge_obj.leaky_data,
        )
        self._submit_evaluation(req, utility)
        _add_evaluation_to_resp(evaluation=utility, resp=resp)


class DataReconstructionPrivacyResource:
    def __init__(
        self,
        challenge_obj: challenge.TabularDataReconstructionChallenge,
        evaluation_submitter: Callable[[falcon.Request, dict[str, float]], None],
    ) -> None:
        self._challenge_obj = challenge_obj
        self._submit_evaluation = evaluation_submitter

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        reconstructed_data = _get_df_from_post_req(req)
        privacy = self._challenge_obj.evaluate_privacy(
            reconstructed_data, self._challenge_obj.leaky_data
        )
        self._submit_evaluation(req, privacy)
        _add_evaluation_to_resp(evaluation=privacy, resp=resp)


def _get_df_from_post_req(req: falcon.Request) -> pd.DataFrame:
    df = utils.bytes_to_dataframe(req.bounded_stream.read())
    return df


def tabular_data_reconstruction_challenge_server_factory(
    challenge_obj: challenge.TabularDataReconstructionChallenge,
) -> falcon.App:
    """A factory to create a web application object which hosts a
    `challenge.TabularDataReconstructionChallenge`.

    By using this factory, you don't have to worry any web-programming issues,
    as they are hidden from you. The generated web app will feature the
    following routes:

    * [GET] `/`
    * [GET] `/data/anonymizer/leaky`
    * [GET] `/data/anonymizer/background-knowledge`
    * [POST] `/utility/anonymizer`
    * [POST] `/utility/deanonymizer`

    Parameters
    ----------
    challenge_obj : challenge.TabularDataReconstructionChallenge
        The data reconstruction challenge to lift to a web application.

    Returns
    -------
    falcon.App
        A web application object based on the falcon web framework, which
        features the supplied `challenge_obj`.
    """
    app = falcon.App()
    app.add_route("/", StaticJSONResource(dict(message="Challenge server is alive!")))
    app.add_route(
        "/data/anonymizer/leaky", StaticDataFrameResource(df=challenge_obj.leaky_data)
    )
    app.add_route(
        "/data/deanonymizer/background-knowledge",
        StaticDataFrameResource(df=challenge_obj.background_knowledge),
    )
    app.add_route(
        "/utility/anonymizer",
        DataReconstructionUtilityResource(
            challenge_obj=challenge_obj,
            # TODO: Change this to a serious submitter
            evaluation_submitter=_demo_anonymizer_submitter,
        ),
    )
    app.add_route(
        "/utility/deanonymizer",
        DataReconstructionPrivacyResource(
            challenge_obj=challenge_obj,
            # TODO: Change this to a serious submitter
            evaluation_submitter=_demo_deanonymizer_submitter,
        ),
    )

    return app
