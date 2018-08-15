Users Guide
===========

Getting Started
---------------

To get started with foreshadow, install the package using pip install. This will also
install the dependencies. Now create a simple python script that uses all the
defaults with Foreshadow.

First import foreshadow

.. code-block:: python

    import foreshadow as fs

Also import sklearn, pandas, and numpy for the demo

.. code-block:: python

    import pandas as pd
    import numpy as np

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
    fs = fs.Foreshadow()
    fs.fit(X_train, y_train)

Now `fs` is a fit Foreshadow object for which all feature engineering has been
performed and the estimator has been trained and optimized. It is now possible to
utilize this exactly as a fit sklearn estimator to make predictions.

.. code-block:: python

    model.score(X_test, y_test)

Great, you now have a working Foreshaow installation! Keep reading to learn how to
export, modify and construct pipelines of your own.


Foreshadow
----------

Foreshadow is the primary object and interface for the Foreshadow framework. By
default, Foreshadow creates a :code:`Preprocessor` object for both the input
data and the target vector.

It also automatically determines whether the target data
is categorical or numerical and determines whether to use a Classification estimator
or a Regressor. By default Foreshadow with either pick `TPOT <https://github.com/EpistasisLab/tpot>`_ for regression or
`auto-sklearn <https://github.com/automl/auto-sklearn>`_ for classification.

**This pipeline is then fit and exposed via the** :code:`fs.pipeline` **object attribute.**

Foreshadow can optionally take in a :py:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`
object for the input data, a :py:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>` object for the target vector, a
:py:obj:`sklearn.base.BaseEstimator` object to fit the preprocessed data, and a :py:obj:`sklearn.grid_search.BaseSearchCV`
class to optimize the available hyperparameters.

Here is an example of a fully defined :py:obj:`Foreshadow <foreshadow.foreshadow.Foreshadow>` object

.. code-block:: python

    shadow = fs.Foreshadow(X_preprocessor=Preprocessor(), y_preprocessor=Preprocessor(), estimator=AutoEstimator(), optimizer=None)

This code is equivalent to the :code:`fs.Foreshadow()` definition but explicitly defines each component. In order to disable one or more
of these components simply pass :code:`False` to the named parameter.

Foreshadow, acting as an estimator is also capable of being used in a :py:obj:`sklearn.pipeline.Pipeline` object. For example:

.. code-block:: python

    pipeline = Pipeline([("estimator", Foreshadow())])
    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test)

By passing an optimizer into Foreshadow, it will attempt to optimize the pipeline it creates by extracting all the hyperparameters from
the preprocessors and the estimator and passing them into the optimizer object along with the partially fit pipeline. This is a potentially
long running process and is not reccomended to be used with estimators such as TPOT or AutoSklearn which also do their own optimization.


Preprocessor
------------

The Preprocessor object provides the feature engineering capabilities for the Foreshadow framework. Like
the :py:obj:`Foreshadow <foreshadow.foreshadow.Foreshadow>` object, the :py:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`
is capable of being used as a standalone object to perform feature engineering, or it can be
used in a :py:obj:`Pipeline <sklearn.pipeline.Pipeline>` as a Transformer to perform preprocessing for an estimator.

In its most-basic form, a Preprocessor can be initialized with no parameters as :code:`fs.Preprocessor()` in which all defaults
will be applied. Ideally, a default preprocessor will be able to produce an acceptable pipeline for feature engineering.

The preprocessor performs the following tasks in order

1. Load configuration (if present)
2. Iterate columns and match Intents
3. Execute single-pipelines on columns in parallel
4. Execute multi-pipelines on columns in series

Intents
~~~~~~~

Preprocessor works by using :py:obj:`Intents <foreshadow.intents.BaseIntent>`. These classes describe a type of feature that a
dataset could possibly contain. For example, we have a :py:obj:`NumericalIntent <foreshadow.intents.NumericalIntent>` and a
:py:obj:`CategoricalIntent <foreshadow.intents.CategoricalIntent>`.

Depending on the characterization of the data performed by the
:code:`is_intent()` class method, *each Intent individually determines if it applies to a particular feature
in the dataset.* However, it is possible for multiple intents to match to a feature. In order to resolve this,
Preprocessor uses a hierarchical structure defined by the superclass (parent) and :code:`children` attributes of
and intent.

This tree-like structure which has :py:obj:`GenericIntent <foreshadow.intents.GenericIntent>` as its
root node is used to prioritize Intents. Intents further down the tree more precisely define a feature, thus the Intent
farthest from the root node that matches a given feature is assigned to it.

Each Intent contains a :code:`multi-pipeline` and a :code:`single-pipeline`. These objects are lists of tuples of the form
:code:`[('name', TransformerObject()),...]` and are used by Preprocessor to construct sklearn Pipeline objects.


Single Pipeline
~~~~~~~~~~~~~~~

A single pipeline operates on a single column of the dataset matched to a specific intent. For example, in the Boston Housing
dataset, the :code:`'CRIM'` column could match to the :py:obj:`NumericalIntent <foreshadow.intents.NumericalIntent>` in which the single pipeline
within that Intent would be executed on that feature.

This process is highly parallelized interally.

Multi Pipeline
~~~~~~~~~~~~~~

Intents also contain a :code:`multi-pipeline` which operates on all columns of data of a given intent simultaneously. For example, in the Boston Housing dataset,
the :code:`'CRIM'` feature (per capita crime rate), the :code:`'RM'` feature (average rooms per house), and the :code:`'TAX'` feature (property tax rate) could be
matched to :py:obj:`NumericIntent <foreshadow.intents.NumericIntent>` in which the corresponding multi-pipline would apply transformers across the columns such as
feature reduction methods like PCA or methods of inference such as Multiple Imputation.

Additionally, while single pipelines are applied on an exclusive basis, multiple pipelines are applied on an inclusive basis. All multiple pipelines in the Intent hierarchy
are executed on matching columns in the order from lowest (most-specific) intent, to the highest (most-general) intent.

**NOTE: All transformers within a single or multi pipeline can access the entire current dataframe as it stands via** :code:`fit_params['full_df']` **in fit or fit_transform**

Smart Transformers
~~~~~~~~~~~~~~~~~~

Smart Transformers are a special subclass of sklearn Transformers derived from the :py:obj:`SmartTransformer <foreshadow.transformers.base.SmartTransformer>` base class.
These transformers do not perform operations on data themselves but instead return a Transformer object at the time of pipeline execution. This allows pipelines to make logical
decisions about actions to perform on features in real-time.

Smart Transformers make up the essence of single and multi pipelines in Intents as they allow conditional operations to be performed on data depending on any statistical analysis
or hypothesis testing. Smart transformers can be overriden using the :code:`override` attribute which takes in a string which is capable of being resolved as an internal transformer
in the Foreshadow library, an external transfomer from sklearn or another smart transformer. The attributes of this override can be set via the :code:`set_params()` methods for which all parameters
other than the :code:`override` parameter itself will be passed to the override object.

To use a smart transformer outside of the Intent / Foreshadow environment simply use it exactly as a sklearn transformer. When you call :code:`fit()` or :code:`fit_transform()` it automatically
resolves which transformer to use by interally calling the :code:`_get_transformer()` overriden method.


Configuration
-------------

The configurability is by far the most powerful aspect of this framework. Through configuration, data scientists can quickly iterate on pipelines generated by Foreshadow and Preprocessor.
Preprocessors take a python dictionary configuration in the :code:`from_json` named parameter in the constructor. This dictionary can be used to override all decision making processes used by
Preprocessor.

An example configuration for processing the Boston Housing dataset is below. We will step through this one by one and demonstrate all the capabilities.

.. code-block:: json

    {
      "columns":{
        "crim":["TestGenericIntent",
                [
                  ["StandardScaler", "Scaler", {"with_mean":false}]
                ]],
        "indus":["TestGenericIntent"]
      },

      "postprocess":[
        ["pca",["age"],[
          ["PCA", "PCA", {"n_components":2}]
        ]]
      ],

      "intents":{
        "TestNumericIntent":{
          "single":[
            ["Imputer", "impute", {"strategy":"mean"}]
          ],
          "multi":[]
        }
      },

    }

The configuration file is composed of a root dictionary containing three hard-coded keys: :code:`columns`,
:code:`postprocess`, and :code:`intents`. First, we will examine the :code:`columns` section.

Column Override
~~~~~~~~~~~~~~~

.. code-block:: json

      "columns":{
        "crim":["TestGenericIntent",
                [
                  ["StandardScaler", "Scaler", {"with_mean":false}]
                ]],
        "indus":["TestGenericIntent"]
      }

This section is a dictionary containing two keys, each of which are columns in the Boston Housing set. First we will look at the value
of the :code:`"crim"` key which is a list.


.. code-block:: json

    ["TestGenericIntent",[
        ["StandardScaler", "Scaler", {"with_mean":false}]
    ]]

The list is of form :code:`[intent_name, pipeline]`. Here we can see that this column has been assigned the intent :code:`"TestGenericIntent`
and the pipeline :code:`[["StandardScaler", "Scaler", {"with_mean":false}]]`

This means that regardless of how Preprocessor automatically assigns Intents, the intent TestGenericIntent will always be assigned to the crim column.
It also means that regardless of what intent is assigned to the column (this value is still important for multi-pipelines), the Preprocessor will always
use this hard coded pipeline to process that column.

The pipeline itself is defined by the following standard :code:`[[class, name, {param_key: param_value, ...}], ...]`
When preprocessor parses this configuration it will create a Pipeline object with the given transformers of the given class, name, and parameters.
For example, the preprocessor above will look something like :code:`sklearn.pipeline.Preprocessor([('Scaler', StandardScaler(with_mean=False)))])`

That pipeline object will be fit on the column crim and will be used to transform it.

Moving on to the :code:`"indus"` column defined by the configuration. We can see that it has an intent override but not a pipeline override. This means
that the default :code:`single_pipeline` for the given intent will be used to process that column.

Intent Override
~~~~~~~~~~~~~~~

.. code-block:: json

    "intents":{
        "TestNumericIntent":{
          "single":[
            ["Imputer", "impute", {"strategy":"mean"}]
          ],
          "multi":[]
        }
    }

Next, we will examine the :code:`intents` section. This section is used to override intents globally, unlike the columns section which overrode intents on a per-column
basis. Any changes to intents defined in this section will apply across the entire Preprocessor pipeline.

The keys in this section each represent the name of an intent. In this example, :code:`TestNumericIntent` is being overridden. The value of the second is a dictionary with the
keys :code:`"single"` and :code:`"multi"` representing the single and multi pipeline override. The value of these pipelines is parsed through the same mechanism is the pipelines
in the columns section.

If a pipeline is empty such as the multi pipeline is above, it will be removed from the final pipeline. However, if the multi key is ommitted from the configuration file, then the default
multi pipeline for that intent will be used.

In this case, for all TestNumericIntent columns, by default, the pipeline :code:`Pipeline([('impute', Imputer(strategy=mean))])` will be executed on the column. No multi-pipeline will be executed
on columns of TestNumericIntent.

Postprocessor Override
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    "postprocess":[
        ["pca",["age"],[
            ["PCA", "PCA", {"n_components":2}]
        ]]
    ]

Finally, in the :code:`postprocess` section of the configuration, you can manually define pipelines to execute on columns of your choosing. The
content of this section is a list of lists of the form :code:`[[name, [cols, ...], pipeline], ...]`. Each list defines a pipeline that will
execute on certain columns. These processes execute after the intent pipelines!

**IMPORTANT** There are two ways of selecting columns through the cols list. By default, specifying a column, or a list of columns



Hyperparameter Tuning
---------------------

Test

