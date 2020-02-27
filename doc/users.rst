.. _users:

User Guide (Under Development)
============

Introduction
============

In this tutorial, we will go through a typical ML workflow with Foreshadow using a subset of the `adult data set <https://archive.ics.uci.edu/ml/datasets/Adult>`_ from the UCI machine learning repository. It is also available under the `example` folder in the foreshadow repository.

Getting Started
===============

To get started with foreshadow, install the package using ``pip install foreshadow``. This will also install the dependencies. Now create a simple python script that uses all the defaults with Foreshadow. Note that Foreshadow requires ``Python >=3.6``.

First import foreshadow related classes. Also import sklearn, pandas and numpy packages.

.. code-block:: python

   from foreshadow import Foreshadow
   from foreshadow.intents import IntentType
   from foreshadow.utils import ProblemType
   from foreshadow.logging import logging

   import numpy as np
   import pandas as pd

   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.linear_model import LinearRegression

Configure the random seed and logging level.

.. code-block:: python

   np.random.seed(42)
   logging.set_level('warning')

Read the Adult Dataset
======================

.. code-block:: python

   data = pd.read_csv('adult.csv').iloc[:2000]
   data.head()

Split data to Train and Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   X_df = data.drop(columns="class")
   y_df = data[["class"]]
   X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2)

Train a Simple LogisticRegression Model using Foreshadow and making predictions
===============================================================================

**The following example is for classification. For regression problems, use ``problem_type=ProblemType.REGRESSION``.**

.. code-block:: python

   shadow = Foreshadow(problem_type=ProblemType.CLASSIFICATION, estimator=LogisticRegression())

.. code-block:: python

   _ = shadow.fit(X_train, y_train)

Making predictions
~~~~~~~~~~~~~~~~~~
.. code-block:: python

   predictions = shadow.predict(X_test)

.. code-block:: python

   predictions.head()

Use the trained estimator to compute the evaluation score.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the scoring method is defined by the selected estimator.

.. code-block:: python

   shadow.score(X_test, y_test)


You can inspect and change Foreshadow's decision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Foreshadow uses a machine learning model to power the auto intent resolving step. As a user, you may not agree with the decision made by Foreshadow. The following APIs allow you to inspect the decisions and change them if you have a different opinion.

.. code-block:: python

   shadow.get_intent('education-num')

If you want to explore a different intent type, simply call the ``override_intent`` API.

.. code-block:: python

   shadow.override_intent('education-num', IntentType.CATEGORICAL)

.. code-block:: python

   _ = shadow.fit(X_train, y_train)

.. code-block:: python

   shadow.score(X_test, y_test)

To show that the intent has been updated:

.. code-block:: python

   shadow.get_intent('education-num')

You can also provide override to fix the intent/column type before fitting
the data. This tells Foreshadow to not run auto intent resolving on some columns but use your decisions instead.

.. code-block:: python

   shadow = Foreshadow(problem_type=ProblemType.CLASSIFICATION, estimator=LogisticRegression())
   shadow.override_intent('education-num', IntentType.CATEGORICAL)
   _ = shadow.fit(X_train, y_train)
   print(shadow.get_intent('education-num'))

Now Let's Search the best Model and Hyper-Parameter
===================================================
At this point, you have a basic pipeline fitted by Foreshadow using a logistic regression estimator. You can update the estimator to something more powerful and retrain the model. Another way is to use the AutoEstimator option in Foreshadow.

Foreshadow leverages the `TPOT AutoML <https://epistasislab.github.io/tpot/using/>`_ package to search the best model and hyper-parameter for you. **Note that AutoML algorithms can take a long time to finish their search, so here we only configure Foreshadow to search for 2 minutes. Please refer to the TPOT manual for more details.**

.. code-block:: python

   from foreshadow.estimators import AutoEstimator
   estimator = AutoEstimator(
       problem_type=ProblemType.CLASSIFICATION,
       auto="tpot",
       estimator_kwargs={"max_time_mins": 2}, # change here
   )
   shadow = Foreshadow(problem_type=ProblemType.CLASSIFICATION, estimator=estimator)

.. code-block:: python

   shadow.override_intent('education-num', IntentType.CATEGORICAL)

.. code-block:: python

   _ = shadow.fit(X_df, y_df)

Making predictions and evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   predictions = shadow.predict(X_test)

.. code-block:: python

   shadow.score(X_test, y_test)

Model persistence
=================

Save the fitted pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After finding the best pipeline, you can export the fitted pipeline as a pickle file for your prediction task.

.. code-block:: python

   pickled_fitted_pipeline_location = "fitted_pipeline.pkl"
   shadow.pickle_fitted_pipeline(pickled_fitted_pipeline_location)

Load back the pipeline for prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pickle

   with open(pickled_fitted_pipeline_location, "rb") as fopen:
       shadow_reload = pickle.load(fopen)

Reuse the pipeline to do predictions and evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   predictions = shadow_reload.predict(X_test)
   predictions.head()

.. code-block:: python

   shadow_reload.score(X_test, y_test)

[Experimental] Register customized data cleaners
================================================

Foreshadow provides several built-in data cleaning transformations. These transformations work on a per column basis.


* datetime cleaner (covert date time into YYYY, mm, and dd respectively)
* financial number cleaner (reformat financial numbers by removing signs like "$" and ",")
* drop cleaner (drop a column if a column has over 90% NaN values)

It is also possible to provide your own data cleaning transformations. The follow (dummy) example shows how to change a column of strings to lowercase.

Define your own cleaner and transformation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two components when defining your own data cleaner (We may change it to only 1 component in the future).


*
  One is the transformation you want to apply to each row in a column.

*
  The second is a subclass of the ``CustomizableBaseCleaner``. You will need to override the ``metric_score`` method. The metric_score returns a confidence score between 0 and 1 representing how certain this particular cleaner should be applied to the column being processed.

.. code-block:: python

   from foreshadow.concrete.internals.cleaners.customizable_base import (
           CustomizableBaseCleaner,
       )

   def lowercase_row(row):
       """Lowercase a row.

       Args:
           row: string of text

       Returns:
           transformed row.

       """
       return row if row is None else str(row).lower()

   class LowerCaseCleaner(CustomizableBaseCleaner):
       def __init__(self):
           super().__init__(transformation=lowercase_row)

       def metric_score(self, X: pd.DataFrame) -> float:
           """Calculate the matching metric score of the cleaner on this col.

           In this method, you specify the condition on when to apply the
           cleaner and calculate a confidence score between 0 and 1 where 1
           means 100% certainty to apply the transformation.

           Args:
               X: a column as a dataframe.

           Returns:
               the confidence score.

           """
           column_name = list(X.columns)[0]
           if column_name == "workclass":
               return 1
           else:
               return 0

Register the cleaner in foreshadow object then train the model
--------------------------------------------------------------

.. code-block:: python

   # Note that right now you need to reinitialize the Foreshadow object before retraining.
   shadow = Foreshadow(problem_type=ProblemType.CLASSIFICATION, estimator=LogisticRegression())
   shadow.register_customized_data_cleaner(data_cleaners=[LowerCaseCleaner])

List the unique values of the workclass column
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   workclass_values = list(X_train["workclass"].unique())
   print(workclass_values)

List the unique values of the workclass after the transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   X_train_cleaned = shadow.X_preparer.steps[0][1].fit_transform(X_train)

   workclass_values_transformed = list(X_train_cleaned["workclass"].unique())
   print(workclass_values_transformed)

Train, predict and evaluate as usual
------------------------------------

.. code-block:: python

   # Note that right now you need to reinitialize the Foreshadow object before retraining.
   shadow = Foreshadow(problem_type=ProblemType.CLASSIFICATION, estimator=LogisticRegression())

   shadow.register_customized_data_cleaner(data_cleaners=[LowerCaseCleaner])

   shadow.fit(X_train, y_train)
   predictions = shadow.predict(X_test)
   shadow.score(X_test, y_test)