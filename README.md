[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Challenge

A library aiding to create challenges for the AnoMed competition platform.

# Usage Example

The following example will create a Falcon-based web app that serves the famous
iris dataset and uses plain binary accuracy as an evaluation metric. In more
detail, these routes are offered (some may have query parameters not mentioned
here):

- [GET] `/` (this displays an "alive message")
- [GET] `/data/anonymizer/training` (this will serve `X_train` and `y_train`)
- [GET] `/data/anonymizer/tuning` (this will serve `X_tune`)
- [GET] `/data/anonymizer/validation` (this will serve `X_val`)
- [GET] `/data/deanonymizer/members` (this will serve a subset of `X_train`
  and `y_train`)
- [GET] `/data/deanonymizer/non-members` (this will serve a subset of `X_val`
  and `y_val`)
- [GET] `/data/attack-success-evaluation` (this will serve data from the
  complement of members and also from the complement of non-members).
- [POST] `/utility/anonymizer` (this will evaluate the quality of an
  anonymizer's prediction compared to `y_tune` or `y_val`.)
- [POST] `/utility/deanonymizer` (this will evaluate the quality of a
  denanonymizer's prediction compared to attack-success-evaluation)

```python
import anomed_challenge as anochal
from sklearn import datasets, model_selection

iris = datasets.load_iris()

X = iris.data  # type: ignore
y = iris.target  # type: ignore

X_train, X_other, y_train, y_other = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_tune, X_val, y_tune, y_val = model_selection.train_test_split(
    X_other, y_other, test_size=0.5, random_state=21
)

example_challenge = anochal.SupervisedLearningMIAChallenge(
    training_data=anochal.InMemoryNumpyArrays(X=X_train, y=y_train),
    tuning_data=anochal.InMemoryNumpyArrays(X=X_tune, y=y_tune),
    validation_data=anochal.InMemoryNumpyArrays(X=X_val, y=y_val),
    anonymizer_evaluator=anochal.strict_binary_accuracy,
    MIA_evaluator=anochal.evaluate_MIA,
    MIA_evaluation_dataset_length=5,
)

# This is what GUnicorn expects
application = anochal.supervised_learning_MIA_challenge_server_factory(
    example_challenge
)
```
