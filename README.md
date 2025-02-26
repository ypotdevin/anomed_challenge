# AnoMed Challenge

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A library aiding to create challenge web servers for the AnoMed competition
platform.

## Preliminaries

The AnoMed platform is basically a network of web servers which use web APIs to
exchange data among each other and provide functionality to each other.
Challenge web servers provide training and evaluation data, which may be
requested via HTTP. They do also offer means to evaluate the utility of
[anonymizers](https://github.com/ypotdevin/anomed_anonymizer) (privacy
preserving machine learning models) via HTTP and means to estimate the privacy
of anonymizers via [attacks](https://github.com/ypotdevin/anomed_deanonymizer)
on them (which we refer to by "deanonymizers" below). Anonymizer web servers
offer input/output access, such that they may be attacked by deanonymizers. For
more details about anonymizers or deanonymizers, view their corresponding
repositories.

In general, you are free to create your own kind of challenge web server, as
long as it offers some well described APIs and follows some general principles,
which we will describe below. You do not need to use this library to submit
challenges. However, if you would like to focus on defining the challenge
itself, without being annoyed by web server related questions, use this library
to generate web servers "for free", which integrate well with the AnoMed
platform.

## How to Create Challenge Web Servers (for selected use cases)

If you goal is to create a challenge that fits one of the following selected
cases, you may use this library's template to create a challenge web server with
minimal effort.

### Supervised Learning Challenge with Membership Inference Attack Threat Model

This scenario assumes that solutions to your challenge (i.e. anonymizers) may be
trained using only a single NumPy feature array `X` (no multiple input arrays)
and a NumPy array of target values `y`. The data is split traditionally into
three parts: training data (for adjustments of weights), tuning data (for
adjustments of the hyperparameters) and validation data (for final evaluation).

The threat model states that membership inference attacks (MIAs) are of interest
and may be used to practically estimate the privacy properties of the
anonymizers, which claim to be privacy preserving machine learning models.
Briefly, MIAs are given a data sample and their goal is to estimate (better than
random guessing), whether that data sample was part of the MIA target's training
data. The MIA's true positive rate at a low false positive rate threshold serves
as an indicator of how well do anonymizers preserve the training data
confidentiality.

MIAs are given a subset of the training data samples (members) and a subset of
validation data samples (non-members) to train, before they will be evaluated.

In the following example, we create a challenge web server (based on the [Falcon
web framework](https://falcon.readthedocs.io/en/stable/)) that serves the famous
iris dataset and uses plain binary accuracy as a anonymizer utility evaluation
metric:

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

The variables `*_train`, `*_tune` and `*_val` contain the challenge data, split
as mentioned above. The custom datatype `InMemoryNumpyArrays` merely bundles
features and targets into one object.

The function `SupervisedLearningMIAChallenge` is the core of this example. It
takes challenge specific parameters to return WSGI compatible Falcon web app,
which which may be utilized by GUnicorn + nginx to create a full-grown web
server. Its arguments `training_data`, `tuning_data` and `validation_data` are
self-explaining. `anonymizer_evaluator` is a function which compares the
validation data target values (ground truth) which an anonymizer's prediction
and returns float value statistics describing the anonymizer's performance.
`MIA_evaluator` is a function which compares the estimated memberships with the
ground truth memberships and returns float value statistics describing the MIA's
performance. `MIA_evaluation_dataset_length` determines the number of members
and also the number of non-members to use for MIA success evaluation (so the
number of samples is twice this value). If possible, set this value to at least 100.

The web app `application` serves these routes:

- [GET] `/` (this displays an "alive message")
- [GET] `/data/anonymizer/training` (this will serve `X_train` and `y_train`)
- [GET] `/data/anonymizer/tuning` (this will serve `X_tune`)
- [GET] `/data/anonymizer/validation` (this will serve `X_val`)
- [GET] `/data/deanonymizer/members` (this will serve a subset of `X_train` and
  `y_train`)
- [GET] `/data/deanonymizer/non-members` (this will serve a subset of `X_val`
  and `y_val`)
- [GET] `/data/attack-success-evaluation` (this will serve data from the
  complement of members and also from the complement of non-members).
- [POST] `/utility/anonymizer` (this will evaluate the quality of an
  anonymizer's prediction compared to `y_tune` or `y_val`.)
- [POST] `/utility/deanonymizer` (this will evaluate the quality of a
  denanonymizer's prediction compared to attack-success-evaluation)

### Dataset Anonymization Challenge with ??? Threat Model

TODO

### Dataset Synthesis Challenge with ??? Threat Model

TODO

## How To Create Challenge Web Servers Without Template

In case your challenge is not covered by one of the available templates, we
suggest that you also use the Falcon web framework and make use of at least some
of the available resource building blocks. Besides that, you should pay
attention to the following principles when implementing your challenge:

- Challenges and submissions will not get any internet access when running on
  the AnoMed platform. Make your challenge self-containing.
- Platform users will not get access to challenge data - only submission
  containers (but not their creators) may access them. That means submission
  contributors have to create their model blueprints »blindly« and are not able
  to have a look into the data when hyperparameter tuning. To make life a little
  easier, we suggest to provide dummy data of the same type and shape as the
  challenge – but with innocuous content outside of the platform. You may post a
  hyperlink to it from within your challenge description.
- Explain your API well in the challenge description, such that custom
  submissions have it easy to obey your API. [Template
  anonymizers](https://github.com/ypotdevin/anomed_anonymizer) and [template
  deanonymizers](https://github.com/ypotdevin/anomed_deanonymizer) are likely
  incompatible with your custom challenge.
- Provide a default route `/` which returns a JSON encoded message like
  "Challenge server is alive!" for diagnosis, upon GET request.
- Challenge data used to fit and evaluate anonymizers or deanonymizers should be
  the same for each submission, to allow for a fair comparison.
- Evaluation data should be disjoint from training data.
- Utility and privacy metrics should be floating point scalars, to allow for
  plotting and ranking. Vectors or even more complex statistics are not suitable
  for that. Also, they should be clearly defined and fixed before the first
  submission comes in. It should not be changed retroactively.
- There should be a way to gain intermediate evaluation results for
  hyperparameter tuning (e.g. with respect to tuning data, if there is any). The
  final evaluation however should be accessed only once by each submission, to
  limit validation data leakage. Such further requests should be rejected.
- Try to find a good compromise between required network capacity and ease of
  use, when sending data via web to submissions. For example, when sending raw
  NumPy arrays over wire, or even plain JSON, no compression is used and usually
  the required network capacity is large. Compressed files on the other hand
  might require further processing in downstream tasks. In the supervised
  learning scenario above for example, we used compressed streams of NumPy
  arrays and [utility functions](https://github.com/ypotdevin/anomed_utils) to
  make working with them comfortable.
