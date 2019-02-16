.. _developers:

Developers Guide
================

Thank you for taking the time to contribute and reading this page, any and all help is appreciated!

Setting up the Project From Source
----------------------------------
General Setup
   1. Clone the project down to your computer
   
      .. code-block:: console
      
         $ git clone https://github.com/georgianpartners/foreshadow.git
         $ cd foreshadow
         $ git checkout development
   
   2. Install and setup `pyenv`_ and `pyenv-virtualenv`_
   
      .. _pyenv: https://github.com/pyenv/pyenv
      .. _pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv
   
      Follow the instructions on their pages or use homebrew if you have a Mac
   
      .. code-block:: console
      
         $ brew install pyenv
         $ brew install pyenv-virtualenv
      
      Make sure to add the following lines to your :code:`.bash_profile`
      
      .. code-block:: bash
      
         export PYENV_ROOT="$HOME/.pyenv"
         export PATH="$PYENV_ROOT/bin:$PATH"
         if command -v pyenv 1>/dev/null 2>&1; then
           eval "$(pyenv init -)"
         fi
         eval "$(pyenv virtualenv-init -)"
      
      Restart your shell session for the changes to take effect and perform the following setup *in the root directory of the project*. This sets up a convenient virtualenv that automatically activates in the root of your project.

      .. code-block:: console
      
         $ pyenv install 3.5.5
         $ pyenv install 3.6.5
         $ pyenv global 3.6.5
         $ pyenv virtualenv -p python3.6 3.6.5 venv
         $ pyenv local venv 3.6.5 3.5.5
   
   3. Install poetry package manager

      .. _poetry: https://github.com/sdispater/poetry
   
      .. code-block:: console
        
         (venv) $ pyenv shell system
         $ curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python 
         $ pyenv shell --unset

Prepare for Autosklearn install
   Autosklearn was setup as an optional dependency as it can be sometimes difficult to install because of its requirement of xgboost. In order to have a development environment that passes all tests, autosklearn is required.

   1. Install swig
   
      Use your package manager to install swig
      
      .. code-block:: console
      
         (venv) $ brew install swig # (or apt-get)
   
   2. Install gcc (MacOS only)
      
      Use your package manager to install gcc (necessary for xgboost)
   
      .. code-block:: console
      
         (venv) $ brew install gcc # (or apt-get)

Install all the packages and commit hooks
   When the project is installed through poetry both project requirements and development requirements are installed. Install commit-hooks using the `pre-commit`_ utility.

   .. _pre-commit: https://pre-commit.com/

   .. code-block:: console
   
      (venv) $ poetry install -v
      (venv) $ poetry run pre-commit install

Making sure everything works
   1. Run pytest to make sure you're good to go
   
      .. code-block:: console
      
         (venv) $ poetry run pytest
   
   2. Run tox to run in supported python versions (optional)
   
      .. code-block:: console
      
         (venv)$ poetry run tox -r # supply the -r flag if you changed the dependencies

   3. Run make html in foreshadow/doc to build the documentation (optional)
   
      .. code-block:: console
      
         (venv) $ poetry run make html
   
   If all the tests pass you're all set up!

Suggested development work flow
   1. Create a branch off of development to contain your change
   
      .. code-block:: console
      
         (venv) $ git checkout development
         (venv) $ git checkout -b {your_feature}

   2. Run pytest and pre-commit while developing
      This will help ensure you haven't horrifically broken something while adding your feature and will help you catch bugs as you develop. Pre-commit will help make sure that your formatting is pristine before create a pull request.
   
      .. code-block:: console
      
         $ poetry run pytest
         $ poetry run pre-commit run --all-files

   3. Run tox to test your changes across versions
      Make sure to add test cases for your change in the appropriate folder in foreshadow/tests and run tox to test your project across python 3.5 and 3.6

      .. code-block:: console
      
         $ poetry run tox

   4. Submit a pull request
      This can be tricky if you have cloned the project instead of forking it but no worries the fix is simple. First go to the project page and **fork it there**. Then do the following.

      .. code-block:: console
      
         (venv) $ git remote add upstream https://github.com/georgianpartners/foreshadow.git
         (venv) $ git remote set-url origin https://github.com/{YOUR_USERNAME}/foreshadow.git
         (venv) $ git push origin {your_feature}
   
      Now you can go to the project on your github page and submit a pull request to the main project. Note, make sure to submit the pull request against the development branch.


Adding Transformers
-------------------

Adding transformers is quite simple. Simply write a class with the `fit` `transform` and `inverse_transform` methods that extends :py:class:`scikit_learn.base.BaseEstimator` and  :py:class:`sklearn.base.TransformerMixin`. Take a look at the structure below and modify it to suit your needs. We would recommend taking a look at the `sklearn.preprocessing.RobustScaler`_ source code for a good example.

.. _sklearn.preprocessing.RobustScaler: https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/preprocessing/data.py#L939

.. code-block:: python

   from sklearn.base import TransformerMixin, BaseEstimator
   from sklearn.utils import check_array
   
   class CustomTransformer(BaseEstimator, TransformerMixin):   
       def fit(self, X, y=None):
           X = check_array(X)
           return self
   
       def transform(self, X, y=None):
           X = check_array(X, copy=True)
           # modify input based on fit here
           return X
   
       def inverse_transform(self, X):
           X = check_array(X, copy=True)
           # if applicable, write inverse transform here
           return X

After writing your transformer make sure place it in the internals folder in its own file with the associated tests for the transformer in the mirrored test directory and you are all set. If you want to add an external transformer that is not already supported by foreshadow submit a pull request with the appropriate modification to the `externals.py` file in transformers.


Adding Smart Transformers
-------------------------

Building smart transformers is even easier than build transformers. Simply extend :py:class:`SmartTransformer <foreshadow.transformers.base.SmartTransformer>` and implement the :py:func:`_get_transformer`. Modify the example below to suit your needs.

.. code-block:: python

   class CustomTransformerSelector(SmartTransformer):
       def _get_transformer(self, X, y=None, **fit_params):
           data = X.iloc[:, 0] # get single column to decide upon
           # perform some computation to determin the best transformer to choose
           return BestTransformer() # return an instance of the selected transformer

Add the smart transformer implementation to the bottom of the `smart.py` file and add the appropriate tests to the mirrored tests folder as well.


Adding Intents
--------------

Intents are where the magic of Foreshadow all comes together. You need to be thoughtful when adding an intent especially with respect to where your intent will slot into the intent tree. This positioning will determine the priority with which the intent is mapped to a column. You will need to subclass your intent off of the parent intent that you determine is the best fit. Intents should be constructed in the form matching :py:class:`BaseIntent <foreshadow.intents.BaseIntent>`.

You will need to set the :py:attr:`dtype <foreshadow.intents.BaseIntent.dtype>`, :py:attr:`children <foreshadow.intents.BaseIntent.children>`, :py:attr:`single_pipeline <foreshadow.intents.BaseIntent.single_pipeline>`, and :py:attr:`multi_pipeline <foreshadow.intents.BaseIntent.multi_pipeline>` class attributes. You will also need to implement the :py:meth:`is_intent <foreshadow.intents.BaseIntent.is_intent>` classmethod. In most cases when adding an intent you can initialize :py:attr:`children <foreshadow.intents.BaseIntent.children>` to an empty list. Set the :py:attr:`dtype <foreshadow.intents.BaseIntent.dtype>` to the most appropriate initial form of that entering your intent.

Use the :py:attr:`single_pipeline <foreshadow.intents.BaseIntent.single_pipeline>` field to determine the transformers that will be applied to a **single** column that is mapped to your intent. Add a **unique** name describing each step that you choose to include in your pipeline. It is important to note the utility of smart transformers here as you can now include branched logic in your pipelines deciding between different individual transformers based on the input data at runtime. The :py:attr:`multi_pipeline <foreshadow.intents.BaseIntent.multi_pipeline>` pipeline should be used to apply transformations to all columns of a specific  intent after the single pipelines have been evaluated. The same rules for defining the pipelines themselves apply here as well.

The :py:meth:`is_intent <foreshadow.intents.BaseIntent.is_intent>` classmethod determines whether a specific column maps to an intent. Use this method to apply any heuristics, logic, or methods of determine whether a raw column maps to the intent that you are defining. Below is an example intent definition that you can modify to suit your needs.

Make **sure** to go to the parent intent and add your intent class name to the ordered :py:attr:`children <foreshadow.intents.BaseIntent.children>` field in the order of priority among the previously defined intents. The last intent in this list will be the most preferred intent upon evaluation in the case of multiple intents being able to process a column.

Take a look at the :py:class:`NumericIntent <foreshadow.intents.NumericIntent>` implementation for an example of how to implement an intent.


Future Architecture Roadmap
---------------------------

Under progress
