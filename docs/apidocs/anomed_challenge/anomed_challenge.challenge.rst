:py:mod:`anomed_challenge.challenge`
====================================

.. py:module:: anomed_challenge.challenge

.. autodoc2-docstring:: anomed_challenge.challenge
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InMemoryNumpyArrays <anomed_challenge.challenge.InMemoryNumpyArrays>`
     - .. autodoc2-docstring:: anomed_challenge.challenge.InMemoryNumpyArrays
          :summary:
   * - :py:obj:`NpzFromDisk <anomed_challenge.challenge.NpzFromDisk>`
     - .. autodoc2-docstring:: anomed_challenge.challenge.NpzFromDisk
          :summary:
   * - :py:obj:`NumpyDataset <anomed_challenge.challenge.NumpyDataset>`
     - .. autodoc2-docstring:: anomed_challenge.challenge.NumpyDataset
          :summary:
   * - :py:obj:`SupervisedLearningMIAChallenge <anomed_challenge.challenge.SupervisedLearningMIAChallenge>`
     - .. autodoc2-docstring:: anomed_challenge.challenge.SupervisedLearningMIAChallenge
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`discard_targets <anomed_challenge.challenge.discard_targets>`
     - .. autodoc2-docstring:: anomed_challenge.challenge.discard_targets
          :summary:
   * - :py:obj:`evaluate_membership_inference_attack <anomed_challenge.challenge.evaluate_membership_inference_attack>`
     - .. autodoc2-docstring:: anomed_challenge.challenge.evaluate_membership_inference_attack
          :summary:
   * - :py:obj:`evaluate_MIA <anomed_challenge.challenge.evaluate_MIA>`
     - .. autodoc2-docstring:: anomed_challenge.challenge.evaluate_MIA
          :summary:
   * - :py:obj:`strict_binary_accuracy <anomed_challenge.challenge.strict_binary_accuracy>`
     - .. autodoc2-docstring:: anomed_challenge.challenge.strict_binary_accuracy
          :summary:

API
~~~

.. py:class:: InMemoryNumpyArrays(X: numpy.ndarray, y: numpy.ndarray)
   :canonical: anomed_challenge.challenge.InMemoryNumpyArrays

   Bases: :py:obj:`anomed_challenge.challenge.NumpyDataset`

   .. autodoc2-docstring:: anomed_challenge.challenge.InMemoryNumpyArrays

   .. rubric:: Initialization

   .. autodoc2-docstring:: anomed_challenge.challenge.InMemoryNumpyArrays.__init__

   .. py:method:: get() -> tuple[numpy.ndarray, numpy.ndarray]
      :canonical: anomed_challenge.challenge.InMemoryNumpyArrays.get

.. py:class:: NpzFromDisk(npz_filepath: str | pathlib.Path, X_label: str = 'X', y_label: str = 'y')
   :canonical: anomed_challenge.challenge.NpzFromDisk

   Bases: :py:obj:`anomed_challenge.challenge.NumpyDataset`

   .. autodoc2-docstring:: anomed_challenge.challenge.NpzFromDisk

   .. rubric:: Initialization

   .. autodoc2-docstring:: anomed_challenge.challenge.NpzFromDisk.__init__

   .. py:method:: get() -> tuple[numpy.ndarray, numpy.ndarray]
      :canonical: anomed_challenge.challenge.NpzFromDisk.get

.. py:class:: NumpyDataset
   :canonical: anomed_challenge.challenge.NumpyDataset

   Bases: :py:obj:`abc.ABC`

   .. autodoc2-docstring:: anomed_challenge.challenge.NumpyDataset

   .. py:method:: get() -> tuple[numpy.ndarray, numpy.ndarray]
      :canonical: anomed_challenge.challenge.NumpyDataset.get
      :abstractmethod:

      .. autodoc2-docstring:: anomed_challenge.challenge.NumpyDataset.get

   .. py:method:: __eq__(other) -> bool
      :canonical: anomed_challenge.challenge.NumpyDataset.__eq__

   .. py:method:: __repr__() -> str
      :canonical: anomed_challenge.challenge.NumpyDataset.__repr__

   .. py:method:: __str__() -> str
      :canonical: anomed_challenge.challenge.NumpyDataset.__str__

.. py:class:: SupervisedLearningMIAChallenge(training_data: anomed_challenge.challenge.NumpyDataset, tuning_data: anomed_challenge.challenge.NumpyDataset, validation_data: anomed_challenge.challenge.NumpyDataset, anonymizer_evaluator: typing.Callable[[numpy.ndarray, numpy.ndarray], dict[str, float]], MIA_evaluator: typing.Callable[[numpy.ndarray, numpy.ndarray], dict[str, float]], MIA_evaluation_dataset_length: int, seed: int | None = None)
   :canonical: anomed_challenge.challenge.SupervisedLearningMIAChallenge

   .. autodoc2-docstring:: anomed_challenge.challenge.SupervisedLearningMIAChallenge

   .. rubric:: Initialization

   .. autodoc2-docstring:: anomed_challenge.challenge.SupervisedLearningMIAChallenge.__init__

   .. py:property:: members
      :canonical: anomed_challenge.challenge.SupervisedLearningMIAChallenge.members
      :type: anomed_challenge.challenge.NumpyDataset

      .. autodoc2-docstring:: anomed_challenge.challenge.SupervisedLearningMIAChallenge.members

   .. py:property:: non_members
      :canonical: anomed_challenge.challenge.SupervisedLearningMIAChallenge.non_members
      :type: anomed_challenge.challenge.NumpyDataset

      .. autodoc2-docstring:: anomed_challenge.challenge.SupervisedLearningMIAChallenge.non_members

   .. py:method:: MIA_evaluation_data(anonymizer: str, deanonymizer: str, data_split: typing.Literal[tuning, validation]) -> tuple[anomed_challenge.challenge.NumpyDataset, numpy.ndarray]
      :canonical: anomed_challenge.challenge.SupervisedLearningMIAChallenge.MIA_evaluation_data

      .. autodoc2-docstring:: anomed_challenge.challenge.SupervisedLearningMIAChallenge.MIA_evaluation_data

   .. py:method:: evaluate_anonymizer(prediction: numpy.ndarray, ground_truth: numpy.ndarray) -> dict[str, float]
      :canonical: anomed_challenge.challenge.SupervisedLearningMIAChallenge.evaluate_anonymizer

      .. autodoc2-docstring:: anomed_challenge.challenge.SupervisedLearningMIAChallenge.evaluate_anonymizer

   .. py:method:: evaluate_membership_inference_attack(prediction: numpy.ndarray, ground_truth: numpy.ndarray) -> dict[str, float]
      :canonical: anomed_challenge.challenge.SupervisedLearningMIAChallenge.evaluate_membership_inference_attack

      .. autodoc2-docstring:: anomed_challenge.challenge.SupervisedLearningMIAChallenge.evaluate_membership_inference_attack

.. py:function:: discard_targets(data: anomed_challenge.challenge.NumpyDataset) -> anomed_challenge.challenge.InMemoryNumpyArrays
   :canonical: anomed_challenge.challenge.discard_targets

   .. autodoc2-docstring:: anomed_challenge.challenge.discard_targets

.. py:function:: evaluate_membership_inference_attack(prediction: numpy.ndarray, ground_truth: numpy.ndarray) -> dict[str, float]
   :canonical: anomed_challenge.challenge.evaluate_membership_inference_attack

   .. autodoc2-docstring:: anomed_challenge.challenge.evaluate_membership_inference_attack

.. py:function:: evaluate_MIA(prediction: numpy.ndarray, ground_truth: numpy.ndarray) -> dict[str, float]
   :canonical: anomed_challenge.challenge.evaluate_MIA

   .. autodoc2-docstring:: anomed_challenge.challenge.evaluate_MIA

.. py:function:: strict_binary_accuracy(prediction: numpy.ndarray, ground_truth: numpy.ndarray) -> dict[str, float]
   :canonical: anomed_challenge.challenge.strict_binary_accuracy

   .. autodoc2-docstring:: anomed_challenge.challenge.strict_binary_accuracy
