.. _users:

User Guide
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

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

Now load in the Boston housing dataset from sklearn into pandas dataframes. This
is a common dataset for testing machine learning models and comes built-in to
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
    shadow = fs.Foreshadow()
    shadow.fit(X_train, y_train)

Now `fs` is a fit Foreshadow object for which all feature engineering has been
performed and the estimator has been trained and optimized. It is now possible to
utilize this exactly as a fit sklearn estimator to make predictions.

.. code-block:: python

    print(shadow.score(X_test, y_test))

Great, you now have a working Foreshaow installation! Keep reading to learn how to
export, modify and construct pipelines of your own.

Here it is all together.

.. code-block:: python

    import foreshadow as fs
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    
    boston = load_boston()
    bostonX_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    bostony_df = pd.DataFrame(boston.target, columns=['target'])
    
    X_train, X_test, y_train, y_test = train_test_split(bostonX_df,
       bostony_df, test_size=0.2)
    shadow = fs.Foreshadow()
    shadow.fit(X_train, y_train)
    
    print(shadow.score(X_test, y_test))


Recommended Workflow
~~~~~~~~~~~~~~~~~~~~

There are many ways to use Foreshadow, but we recommend using this workflow initially as it is the quickest and easiest way to
generate a high-performing model with minimum effort.

First, prep your data into X_train, X_test, y_train and y_test pandas dataframes. For example:

.. code-block:: python

    boston = load_boston()
    bostonX_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    bostony_df = pd.DataFrame(boston.target, columns=['target'])

    X_train, X_test, y_train, y_test = train_test_split(bostonX_df,
       bostony_df, test_size=0.2)


Then initialize a default Foreshadow object with a sklearn estimator such as XGBoost. We want this
process to be fast so we can iterate, so for the time being we will override the default TPOT model selection,
ensembling and hyperparameter optimization for regression problems with a simple default XGBoost regressor.

.. code-block:: python

    shadow = fs.Foreshadow(estimator=XGBRegressor())

Then fit the train data on that object

.. code-block:: python

    shadow.fit(X_train, Y_train)

You now have an initial pipeline. Lets see how it did and serialize it to a JSON file so we can look at it.

.. code-block:: python

    # Score the pipeline
    shadow.score(X_test, y_test)

    # Serialize the pipeline
    x_proc = shadow.X_preparer.serialize()
    y_proc = shadow.y_preparer.serialize()

    # Write the serialized pipelines to file
    json.dump(x_proc, open("x_proc.json", "w"), indent=2)
    json.dump(y_proc, open("y_proc.json", "w"), indent=2)

Now we have two pipeline configurations, one for our X data and one for our Y data. We also have an initial idea
of how well the initial pipeline performed.

Let's suppose that you want to experiment with a different scaler. Open the configuration JSON and make this change. (Look to the `Configuration`_ section for more details on this)

For example, add the following snippet to the bottom or x_proc.json

.. code-block:: python

    "combinations": [
        {
          "columns.CHAS.0": "['NumericIntent', 'CategoricalIntent']"
        }
     ]

Now let's re-create the Foreshadow object with your changes.

.. code-block:: python

    import json

    # Load in the configs from file
    x_proc = json.load(open("x_proc.json", "r"))
    y_proc = json.load(open("y_proc.json", "r"))

    # Create the preprocessors
    x_processor = fs.Preprocessor(from_json=x_proc)
    y_processor = fs.Preprocessor(from_json=y_proc)

    # Create the foreshadow object
    shadow = fs.Foreshadow(X_preparer=x_processor, y_preparer=y_processor, estimator=XGBRegressor())

    # Fit the foreshadow object
    shadow.fit(X_train, y_train)

    # Score the foreshadow object
    shadow.score(X_test, y_test)

Now we can see the performance difference as a result of the changes. This process of swapping in and out different scalers is slow and tedious though. Let's add a combinations section to the configuration file and let an optimizer do the heavy lifting of evaluating the framework.

First, read the `Hyperparameter Tuning`_ section about how hyperparameter optimization works in Foreshadow. Then add a combinations section to the exported JSON file(s) you have from the preprocessor. Remember that the more parameters you add, the longer it will take. We recommend focusing on a set of related parameters one by one and optimizing them individually. e.g. Optimize thresholds for Scaling, then thresholds for Encoding, then feature reduction (PCA / LDA) etc.

Once you add a combinations section to figure out the best parameters, create the Foreshadow object again, except this time with an optimizer such as GridSearchCV or RandomSearchCV from sklearn.

.. code-block:: python

    # Load in the configs from file
    x_proc_combo = json.load(open("x_proc_combo.json", "r"))
    y_proc_combo = json.load(open("y_proc_combo.json", "r"))

    # Create the preprocessors
    x_processor = Preprocessor(from_json=x_proc_combo)
    y_processor = Preprocessor(from_json=y_proc_combo)

    # Create the foreshadow object
    shadow = fs.Foreshadow(X_preparer=x_processor, y_preparer=y_processor, estimator=XGBRegressor(), optimizer=GridSearchCV)

    # Fit the foreshadow object
    shadow.fit(X_train, y_train)

    # Score the foreshadow object
    shadow.score(X_test, y_test)

    # Extract the optimized pipeline
    pipeline = shadow.pipeline

    # Save it to file
    pickle.dump(pipeline, open("final_pipeline.pkl", "wb"))

    # Export the best pipelines

    # Serialize the pipeline
    x_proc_best = shadow.X_preparer.serialize()
    y_proc_best = shadow.y_preparer.serialize()

    # Write the serialized pipelines to file
    json.dump(x_proc_best, open("x_proc_best.json", "w"), indent=2)
    json.dump(y_proc_best, open("y_proc_best.json", "w"), indent=2)


Once you have a preprocessor pipeline that you are happy with, you should attempt to optimize the model. The AutoEstimator will be good for this as it will automatically do model selection and hyperparameter optimization. To do this, construct the Foreshadow object in the same way as above, using the optimized JSON configuration, but instead of passing in an sklearn estimator and optimizer, leave those fields as default. This will force Foreshadow to use the defaults which automatically chooses either TPOT (regression) or AutoSklearn (classification) to fit the preprocessed data without any of their in-built feature engineering. When serializing the pipeline, Foreshadow will automatically choose the pipeline with the highest cross-validation score.

*This will take a long time to execute... get yourself a cup of coffee or tea, sit back, and relax*

Great! Now you have an optimized sklearn pipeline that you can share, load, manipulate, and inspect!


Foreshadow
----------

Foreshadow is the primary object and interface for the Foreshadow framework. By
default, Foreshadow creates a :code:`Preprocessor` object for both the input
data and the target vector.

It also automatically determines whether the target data
is categorical or numerical and determines whether to use a Classification estimator
or a Regressor. By default Foreshadow will either pick `TPOT <https://github.com/EpistasisLab/tpot>`_ for regression or
`auto-sklearn <https://github.com/automl/auto-sklearn>`_ for classification.

**This pipeline is then fit and exposed via the** :code:`pipeline` **object attribute.**

Foreshadow can optionally take in a :py:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`
object for the input data, a :py:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>` object for the target vector, a
:py:obj:`sklearn.base.BaseEstimator` object to fit the preprocessed data, and a :py:obj:`sklearn.grid_search.BaseSearchCV`
class to optimize the available hyperparameters.

Here is an example of a fully defined :py:obj:`Foreshadow <foreshadow.foreshadow.Foreshadow>` object

.. code-block:: python

    shadow = fs.Foreshadow(X_preparer=Preprocessor(), y_preparer=Preprocessor(), estimator=AutoEstimator(), optimizer=None)

This code is equivalent to the :code:`fs.Foreshadow()` definition but explicitly defines each component. In order to disable one or more
of these components simply pass :code:`False` to the named parameter (Note that the default :code:`None` automatically initializes the above).

:py:obj:`AutoEstimator <foreshadow.estimators.AutoEstimator>` is automatically defined as the estimator for Foreshadow. This estimator detects the problem type (classification or regression)
and then either uses TPOT or Auto-Sklearn to serve as the estimator. The preprocessing methods are stripped from TPOT and Auto-Sklearn when they are used in this manner as we favor our own
Preprocessor over their methods. As such these two frameworks will only perform model selection and estimator hyperparameter optimization by default.

**NOTE:** Future work includes implementing TPOT and AutoSklean's optimizers into this platform such that they can be used for both model selection and optimizing hyperparameters for the feature
engineering aspects. Until then, however, they will only optimize the model as they are blind to the earlier parts of the pipeline.

Foreshadow, acting as an estimator is also capable of being used in a :py:obj:`sklearn.pipeline.Pipeline` object. For example:

.. code-block:: python

    pipeline = Pipeline([("estimator", Foreshadow())])
    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test)

By passing an optimizer into Foreshadow, it will attempt to optimize the pipeline it creates by extracting all the hyperparameters from
the preprocessors and the estimator and passing them into the optimizer object along with the partially fit pipeline. This is a potentially
long-running process and is not reccomended to be used with estimators such as TPOT or AutoSklearn which also do their own optimization.


Preprocessor
------------

The Preprocessor object provides the feature engineering capabilities for the Foreshadow framework. Like
the :py:obj:`Foreshadow <foreshadow.foreshadow.Foreshadow>` object, the :py:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`
is capable of being used as a standalone object to perform feature engineering, or it can be
used in a :py:obj:`Pipeline <sklearn.pipeline.Pipeline>` as a Transformer to perform preprocessing for an estimator.

In its most basic form, a Preprocessor can be initialized with no parameters as :code:`fs.Preprocessor()` in which all defaults
will be applied. Ideally, a default preprocessor will be able to produce an acceptable pipeline for feature engineering.

The preprocessor performs the following tasks in order

1. Load configuration (if present)
2. Iterate columns and match Intents
3. Execute single-pipelines on columns in parallel
4. Execute multi-pipelines on columns in series

Intents
~~~~~~~

Preprocessor works by using :py:obj:`Intents <foreshadow.intents.BaseIntent>`. These classes describe a type of feature that a
dataset could possibly contain. For example, we have a :py:obj:`NumericIntent <foreshadow.intents.NumericIntent>` and a
:py:obj:`CategoricalIntent <foreshadow.intents.CategoricalIntent>`.

Depending on the characterization of the data performed by the
:code:`is_intent()` class method, *each Intent individually determines if it applies to a particular feature
in the dataset.* However, it is possible for multiple intents to match to a feature. In order to resolve this,
Preprocessor uses a hierarchical structure defined by the superclass (parent) and :code:`children` attributes of
and intent. There is also a priority order defined in each intent to break ties at the same level.

This tree-like structure which has :py:obj:`GenericIntent <foreshadow.intents.GenericIntent>` as its
root node is used to prioritize Intents. Intents further down the tree more precisely define a feature and intents further to the right hold a higher priority than those to the left, thus the Intent represented by the right-most node of the tree that matches will be selected.

Each Intent contains a :code:`multi-pipeline` and a :code:`single-pipeline`. These objects are lists of tuples of the form
:code:`[('name', TransformerObject()),...]` and are used by Preprocessor to construct sklearn Pipeline objects.


Single Pipeline
~~~~~~~~~~~~~~~

The single pipeline defines operations (transformations of data) on a single column of the dataset matched to a specific intent. For example, in the Boston Housing
dataset, the :code:`'CRIM'` column could match to the :py:obj:`NumericIntent <foreshadow.intents.NumericIntent>` in which the single pipeline
within that Intent would be executed on that feature.

This process is highly parallelized interally.

Multi Pipeline
~~~~~~~~~~~~~~

Intents also contain a :code:`multi-pipeline` which operates on all columns of data of a given intent simultaneously. For example, in the Boston Housing dataset,
the :code:`'CRIM'` feature (per capita crime rate), the :code:`'RM'` feature (average rooms per house), and the :code:`'TAX'` feature (property tax rate) could be
matched to :py:obj:`NumericIntent <foreshadow.intents.NumericIntent>` in which the corresponding multi-pipeline would apply transformers across the columns such as
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

.. note:: Arguments passed into the constructor of a smart transformer will be passed into the fit function of the transformer it resolves to. This is meant to primarily be used alongside the override argument.


Configuration
-------------

The configurability is by far the most powerful aspect of this framework. Through configuration, data scientists can quickly iterate on pipelines generated by Foreshadow and Preprocessor.
Preprocessors take a python dictionary configuration in the :code:`from_json` named parameter in the constructor. This dictionary can be used to override all decision -making processes used by
Preprocessor.

An example configuration for processing the Boston Housing dataset is below. We will step through this one by one and demonstrate all the capabilities.

.. code-block:: json

    {
        "columns": {
            "crim": {
                "intent": "GenericIntent",
                "pipeline": [{
                    "transformer": "StandardScaler",
                    "name": "Scaler",
                    "parameters": {
                        "with_mean": false
                    }
                }]
            },
            "indus": {
                "intent": "GenericIntent"
            }
        },
        "postprocess": [{
            "name": "pca",
            "columns": ["age"],
            "pipeline": [{
                "transformer": "PCA",
                "name": "PCA",
                "parameters": {
                    "n_components": 2
                }
            }]
        }],
        "intents": {
            "NumericIntent": {
                "single": [{
                    "transformer": "Imputer",
                    "name": "impute",
                    "parameters": {
                        "strategy": "mean"
                    }
                }],
                "multi": []
            }
        }
    }

The configuration file is composed of a root dictionary containing three hard-coded keys: :code:`columns`,
:code:`postprocess`, and :code:`intents`. First, we will examine the :code:`columns` section.

Column Override
~~~~~~~~~~~~~~~

.. code-block:: python

    "columns": {
        "crim": {
            "intent": "GenericIntent",
            "pipeline": [{
                "transformer": "StandardScaler",
                "name": "Scaler",
                "parameters": {
                    "with_mean": false
                }
            }]
        },
        "indus": {
            "intent": "GenericIntent"
        }
    }

This section is a dictionary containing two keys, each of which are columns in the Boston Housing set. First we will look at the value
of the :code:`"crim"` key which is a dict.


.. code-block:: json

    {
        "intent": "GenericIntent",
        "pipeline": [{
            "transformer": "StandardScaler",
            "name": "Scaler",
            "parameters": {
                "with_mean": false
            }
        }]
    }

Here we can see that this column has been assigned the intent :code:`"GenericIntent`
and the pipeline :code:`[{"transformer": "StandardScaler", "name": "Scaler", "parameters": {"with_mean":false}}]`

This means that regardless of how Preprocessor automatically assigns Intents, the intent GenericIntent will always be assigned to the crim column.
It also means that regardless of what intent is assigned to the column (this value is still important for multi-pipelines), the Preprocessor will always
use this hard-coded pipeline to process that column. The column would still be processed by its initially identifited multi-pipeline unless explicitly overridden.

The pipeline itself is defined by the following standard :code:`[{"transformer":class, "name":name, "parameters":{param_key: param_value, ...}], ...]`
When preprocessor parses this configuration it will create a Pipeline object with the given transformers of the given class, name, and parameters.
For example, the preprocessor above will look something like :code:`sklearn.pipeline.Preprocessor([('Scaler', StandardScaler(with_mean=False)))])`
Any class implementing the sklearn Transformer standard (including SmartTransformer) can be used here.

That pipeline object will be fit on the column crim and will be used to transform it.

Moving on to the :code:`"indus"` column defined by the configuration. We can see that it has an intent override but not a pipeline override. This means
that the default :code:`single_pipeline` for the given intent will be used to process that column. By default the serialized pipeline will have
a list of partially matching intents under the "all_matched_intents" dict key. These can likely be substituted into the Intent name with little or no
compatibility issues.

Intent Override
~~~~~~~~~~~~~~~

.. code-block:: python

    "intents": {
        "NumericIntent": {
            "single": [{
                "transformer": "Imputer",
                "name": "impute",
                "parameters": {
                    "strategy": "mean"
                }
            }],
            "multi": []
        }
    }


Next, we will examine the :code:`intents` section. This section is used to override intents globally, unlike the columns section which overrode intents on a per-column
basis. Any changes to intents defined in this section will apply across the entire Preprocessor pipeline. However, individual pipelines defined in the columns section will override pipelines defined here.

The keys in this section each represent the name of an intent. In this example, :code:`NumericIntent` is being overridden. The value is a dictionary with the
keys :code:`"single"` and :code:`"multi"` respresent the single and multi pipeline overrides. The value of these pipelines is parsed through the same mechanism as the pipelines
in the columns section.

If a pipeline is empty such as the multi pipeline is above, it will be removed from the final pipeline. However, if the multi key is ommitted from the configuration file, then the default
multi pipeline for that intent will be used.

In this case, for all NumericIntent columns, by default, the pipeline :code:`Pipeline([('impute', Imputer(strategy=mean))])` will be executed on the column. No multi-pipeline will be executed
on columns of NumericIntent.

Postprocessor Override
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "postprocess": [{
            "name": "pca",
            "columns": ["age"],
            "pipeline": [{
                "class": "PCA",
                "name": "PCA",
                "parameters": {
                    "n_components": 2
                }
            }]
        }]
    }

Finally, in the :code:`postprocess` section of the configuration, you can manually define pipelines to execute on columns of your choosing. The
content of this section is a list of dictionaries of the form :code:`[{"name":name, "columns":[cols, ...], "pipeline":pipeline}, ...]`. Each list defines a pipeline that will
execute on certain columns. These processes execute after the intent pipelines!

**IMPORTANT** There are two ways of selecting columns through the cols list. By default, specifying a column, or a list of columns, will automatically select
the columns in the data frame that are computed columns deriving from that column. For example, in the list above, all columns derived from the :code:`age` column
will be passed into the PCA transformer and reduced to 2 components. To override this behavior and select columns by their name at the current stage in the process,
prepend a dollar sign to the column name. For example :code:`["$age_scale_0", "$indus_encode_0", "$indus_encode_1"]`


Through overriding these various components, any combination of feature engineering can be achieved. To generate this configuration dictionary after fitting a Preprocessor or a
Foreshadow object, run the :code:`serialize()` method on the Preprocessor object or on :code:`Foreshadow.X_preparer` or :code:`y_preparer`. That dictionary can be programmatically modified in python
or can be serialized to JSON where it can be modified by hand. By default the output of :code:`serialize()` will fix all
feature engineering to be constant. To only enforce sections of the configuration output from :code:`serialize()` simply copy and paste the relevant sections into a new JSON file.



Hyperparameter Tuning
---------------------

Foreshadow also supports hyperparameter tuning through two mechanisms. By default, Foreshadow will use :py:obj:`AutoEstimator <foreshadow.estimators.AutoEstimator>` as an estimator
in the pipeline. This estimator will automatically choose either TPOT, for regression problems or AutoSklearn for classification problems. It also strips all feature engineering and preprocessing
from these two frameworks. This, in effect, uses TPOT and AutoSklearn only for model selection and model hyperparameter optimization. These estimators are not passed hyperparameters from the Preprocessor
and thus will not optimize them.

The second method of hyperparameter tuning is to use a vanilla sklearn estimator when declaring foreshadow (such as XGBoost or LogisticRegression) and also pass in a :py:obj:`BaseSearchCV <sklearn.grid_search.BaseSearchCV>`
class into the :code:`optimizer` parameter. This will use the provided optimizer to perform a parameter search on both the preprocessing and the model at the same time. The parameter search space for this configuration is defined
in two locations.

Default Dictionary
~~~~~~~~~~~~~~~~~~

The first is in :code:`foreshadow/optimizers/param_mapping.py` which contains a dictionary like:

.. code-block:: python

    config_dict = {
        "StandardScaler.with_std": [True, False]
        "StandardScaler.with_mean": [True, False]
        }

This dictionary contains keys and values of the form :code:`ClassName.attribute: iterator(test_values)` If any items in the pipeline match the classname.attribute selector then that attribute will be added as a
hyperparameter with the values of the iterator (list, generator, etc.) as the search space.

**NOTE:** In the future, this dictionary will be able to be passed in to Foreshadow, for now it must be modified manually if changes wish to be made.

JSON Combinations Config
~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to manually define spaces to search for the Preprocessor those can be defined in the configuration dictionary of the preprocessor in the :code:`combinations` section.
This is what a combinations section looks like.

.. code-block:: json

    {
        "columns": {
            "crim": {
                "intent": "GenericIntent",
                "pipeline": [{
                    "transformer": "StandardScaler",
                    "name": "Scaler",
                    "parameters": {
                        "with_mean": false
                    }
                }]
            },
            "indus": {
                "intent": "GenericIntent"
            }
        },
    
        "postprocess": [],
    
        "intents": {},
    
        "combinations": [{
            "columns.crim.pipeline.0.parameters.with_mean": "[True, False]",
            "columns.crim.pipeline.0.name": "['Scaler', 'SuperScaler']"
        }]
    
    }



This section of the configuration file is a list of dictionaries. Each dictionary represents a single parameter space definition that should be searched. Within these dictionaries
each key is an identifier for a value in another part of the configuration file. For example :code:`columns.crim.1.0.2.with_mean` will identify the *columns* key and then the *crim* key, then
the 1th index of that list, the 0th index of the next list, the 2nd index of the next list, and finally the *with_mean* key of that dictionary. Each value is a string of **python code** that
will be evaluated to create an **iterator** object that will be used to generate the parameter space.

In this example 4 combinations will be searched:

* :code:`StandardScaler(with_mean=False, name="Scaler")`
* :code:`StandardScaler(with_mean=True, name="Scaler")`
* :code:`StandardScaler(with_mean=False, name="SuperScaler")`
* :code:`StandardScaler(with_mean=True, name="SuperScaler")`

*In addition to any search parameters defined in the default search space dictionary above*

