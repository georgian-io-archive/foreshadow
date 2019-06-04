.. _index:
.. foreshadow documentation master file, created by
   sphinx-quickstart on Thu Aug  9 11:43:44 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Foreshadow: Simple Machine Learning Scaffolding
===============================================
|License| |BuildStatus| |Coverage| |Code style: black|

**Foreshadow** is a one of a kind solution to the mess that is machine learning pipelines.

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://github.com/georgianpartners/foreshadow/blob/master/LICENSE
.. |BuildStatus| image:: https://travis-ci.org/georgianpartners/foreshadow.svg?branch=master
   :target: https://travis-ci.org/georgianpartners/foreshadow
.. |Coverage| image:: https://coveralls.io/repos/github/georgianpartners/foreshadow/badge.svg?branch=development
   :target: https://coveralls.io/github/georgianpartners/foreshadow
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

------------------------

**Grab a cup of coffee, relax, and let foreshadow build pipelines for you**

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.datasets import load_boston
   from sklearn.model_selection import train_test_split
   import foreshadow as fs

   np.random.seed(0)
   boston = load_boston()
   bostonX_df = pd.DataFrame(boston.data, columns=boston.feature_names)
   bostony_df = pd.DataFrame(boston.target, columns=['target'])
   X_train, X_test, y_train, y_test = train_test_split(bostonX_df, 
       bostony_df, test_size=0.2)

   model = fs.Foreshadow()
   model.fit(X_train, y_train)
   model.score(X_test, y_test)


About
-----
Foreshadow is an automatic pipeline generation tool that makes creating, iterating,
and evaluating machine learning pipelines a fast and intuitive experience allowing
data scientists to spend more time on data science and less time on code.


Key Features
------------
- Automatic Feature Engineering
- Automatic Model Selection
- Rapid Pipeline Development / Iteration
- Automatic Parameter Optimization
- Ease of Extensibility
- Scikit-Learn Compatible

Foreshadow supports python 3.6+


The User Guide
--------------
.. toctree::
   :maxdepth: 2

   users

.. toctree::
   :maxdepth: 1
   :hidden:

   faq


The Developer Guide
-------------------
.. toctree::
   :maxdepth: 2

   developers

.. toctree::
   :maxdepth: 1
   :hidden:

   contrib

API
---
.. toctree::
   :maxdepth: 2

   api

.. toctree::
   :maxdepth: 1

   architecture

Changelog
---------
.. toctree::
   :maxdepth: 2

   changelog

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
