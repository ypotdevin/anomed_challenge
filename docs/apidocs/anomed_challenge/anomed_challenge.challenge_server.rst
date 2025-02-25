:py:mod:`anomed_challenge.challenge_server`
===========================================

.. py:module:: anomed_challenge.challenge_server

.. autodoc2-docstring:: anomed_challenge.challenge_server
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DynamicNumpyDataResource <anomed_challenge.challenge_server.DynamicNumpyDataResource>`
     - .. autodoc2-docstring:: anomed_challenge.challenge_server.DynamicNumpyDataResource
          :summary:
   * - :py:obj:`StaticJSONResource <anomed_challenge.challenge_server.StaticJSONResource>`
     - .. autodoc2-docstring:: anomed_challenge.challenge_server.StaticJSONResource
          :summary:
   * - :py:obj:`StaticNumpyDataResource <anomed_challenge.challenge_server.StaticNumpyDataResource>`
     - .. autodoc2-docstring:: anomed_challenge.challenge_server.StaticNumpyDataResource
          :summary:
   * - :py:obj:`UtilityResource <anomed_challenge.challenge_server.UtilityResource>`
     - .. autodoc2-docstring:: anomed_challenge.challenge_server.UtilityResource
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`supervised_learning_MIA_challenge_server_factory <anomed_challenge.challenge_server.supervised_learning_MIA_challenge_server_factory>`
     - .. autodoc2-docstring:: anomed_challenge.challenge_server.supervised_learning_MIA_challenge_server_factory
          :summary:

API
~~~

.. py:class:: DynamicNumpyDataResource(individual_data_provider: typing.Callable[[falcon.Request], anomed_challenge.challenge.NumpyDataset])
   :canonical: anomed_challenge.challenge_server.DynamicNumpyDataResource

   .. autodoc2-docstring:: anomed_challenge.challenge_server.DynamicNumpyDataResource

   .. rubric:: Initialization

   .. autodoc2-docstring:: anomed_challenge.challenge_server.DynamicNumpyDataResource.__init__

.. py:class:: StaticJSONResource(obj: typing.Any)
   :canonical: anomed_challenge.challenge_server.StaticJSONResource

   .. autodoc2-docstring:: anomed_challenge.challenge_server.StaticJSONResource

   .. rubric:: Initialization

   .. autodoc2-docstring:: anomed_challenge.challenge_server.StaticJSONResource.__init__

.. py:class:: StaticNumpyDataResource(_data: anomed_challenge.challenge.NumpyDataset)
   :canonical: anomed_challenge.challenge_server.StaticNumpyDataResource

   .. autodoc2-docstring:: anomed_challenge.challenge_server.StaticNumpyDataResource

   .. rubric:: Initialization

   .. autodoc2-docstring:: anomed_challenge.challenge_server.StaticNumpyDataResource.__init__

.. py:class:: UtilityResource(target_data_provider: typing.Callable[[falcon.Request], numpy.ndarray], evaluator: typing.Callable[[numpy.ndarray, numpy.ndarray], dict[str, float]], submitter: typing.Callable[[falcon.Request, dict[str, float]], None])
   :canonical: anomed_challenge.challenge_server.UtilityResource

   .. autodoc2-docstring:: anomed_challenge.challenge_server.UtilityResource

   .. rubric:: Initialization

   .. autodoc2-docstring:: anomed_challenge.challenge_server.UtilityResource.__init__

   .. py:method:: on_post(req: falcon.Request, resp: falcon.Response) -> None
      :canonical: anomed_challenge.challenge_server.UtilityResource.on_post

      .. autodoc2-docstring:: anomed_challenge.challenge_server.UtilityResource.on_post

.. py:function:: supervised_learning_MIA_challenge_server_factory(challenge_obj: anomed_challenge.challenge.SupervisedLearningMIAChallenge) -> falcon.App
   :canonical: anomed_challenge.challenge_server.supervised_learning_MIA_challenge_server_factory

   .. autodoc2-docstring:: anomed_challenge.challenge_server.supervised_learning_MIA_challenge_server_factory
