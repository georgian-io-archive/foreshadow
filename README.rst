Foreshadow: Simple Machine Learning Scaffolding
===============================================

|BuildStatus| |DocStatus| |Coverage| |CodeStyle| |License|

Foreshadow is an automatic pipeline generation tool that makes creating, iterating,
and evaluating machine learning pipelines a fast and intuitive experience allowing
data scientists to spend more time on data science and less time on code.

.. |BuildStatus| image:: https://dev.azure.com/georgianpartners/foreshadow/_apis/build/status/georgianpartners.foreshadow?branchName=master
   :target: https://dev.azure.com/georgianpartners/foreshadow/_build/latest?definitionId=1&branchName=master

.. |DocStatus| image:: https://readthedocs.org/projects/foreshadow/badge/?version=latest
  :target: https://foreshadow.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. |Coverage| image:: https://img.shields.io/azure-devops/coverage/georgianpartners/foreshadow/1.svg
  :target: https://dev.azure.com/georgianpartners/foreshadow/_build/latest?definitionId=1&branchName=master
  :alt: Coverage

.. |CodeStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/ambv/black
  :alt: Code Style

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
  :target: https://github.com/georgianpartners/foreshadow/blob/master/LICENSE
  :alt: License

Key Features
------------
- Scikit-Learn compatible
- Automatic column intent inference (currently supports Numerical and Categorical Types)
- Allow user override on column intent and transformation functions
- Automatic feature preprocessing depending on the column intent type
- Automatic model selection
- Rapid pipeline development / iteration

Features in the road map
------------------------
- Automatic column intent inference for DateTime, Text and Droppable types
- Automatic feature engineering
- Automatic parameter optimization

Foreshadow supports python 3.6+

Installing Foreshadow
---------------------

.. code-block:: console

    $ pip install foreshadow

Read the documentation to `set up the project from source`_.

.. _set up the project from source: https://foreshadow.readthedocs.io/en/development/developers.html#setting-up-the-project-from-source

Getting Started
---------------

To get started with foreshadow, install the package using pip install. This will also
install the dependencies. Now create a simple python script that uses all the
defaults with Foreshadow.

First import foreshadow

.. code-block:: python

    from foreshadow.foreshadow import Foreshadow
    from foreshadow.estimators import AutoEstimator
    from foreshadow.utils import ProblemType

Also import sklearn, pandas, and numpy for the demo

.. code-block:: python

    import pandas as pd

    from sklearn.datasets import boston_housing
    from sklearn.model_selection import train_test_split

Now load in the boston housing dataset from sklearn into pandas dataframes. This
is a common dataset for testing machine learning models and comes built in to
scikit-learn.

.. code-block:: python

    boston = load_boston()
    bostonX_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    bostony_df = pd.DataFrame(boston.target, columns=['target'])

Next, exactly as if working with an sklearn estimator, perform a train test
split on the data and pass the train data into the fit function of a new Foreshadow
object

.. code-block:: python

    X_train, X_test, y_train, y_test = train_test_split(bostonX_df,
       bostony_df, test_size=0.2)

    problem_type = ProblemType.REGRESSION

    estimator = AutoEstimator(
        problem_type=problem_type,
        auto="tpot",
        estimator_kwargs={"max_time_mins": 1},
    )
    shadow = Foreshadow(estimator=estimator, problem_type=problem_type)
    shadow.fit(X_train, y_train)

Now `fs` is a fit Foreshadow object for which all feature engineering has been
performed and the estimator has been trained and optimized. It is now possible to
utilize this exactly as a fit sklearn estimator to make predictions.

.. code-block:: python

    shadow.score(X_test, y_test)

Great, you now have a working Foreshaow installation! Keep reading to learn how to
export, modify and construct pipelines of your own.

Tutorial
------------
We also have a jupyter notebook tutorial to go through more details under the `examples` folder.

Documentation
-------------
`Read the docs!`_

.. _Read the docs!: https://foreshadow.readthedocs.io/en/development/index.html
