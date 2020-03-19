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
      
      Restart your shell session for the changes to take effect and perform the following setup **in the root directory of the project**. This sets up a convenient virtualenv that automatically activates in the root of your project. (Note: there is a `known error with pyenv <https://github.com/pyenv/pyenv/issues/1219#issuecomment-487206619>`_. Also, you may need to change the file path depending on your version or you may not even need to do that step.

      .. code-block:: console
         
         $ open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
         $ pyenv install 3.6.8
         $ pyenv global 3.6.8
         $ pyenv virtualenv -p python3.6 3.6.8 venv
         $ pyenv local venv 3.6.8
   
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
      
         (venv) $ brew install gcc@5 # (or apt-get)

Install all the packages and commit hooks
   When the project is installed through poetry both project requirements and development requirements are installed. Install commit-hooks using the `pre-commit`_ utility.

   .. _pre-commit: https://pre-commit.com/

   .. code-block:: console

      (venv) $ poetry install -v
      (venv) $ export CC=gcc-5; export CXX=g++-5;
      (venv) $ poetry install -E dev
      (venv) $ poetry run pre-commit install

Configure PlantUML

  .. code-block:: console

  (venv) $ brew install plantuml # MacOS (requires brew cask install adoptopenjdk)
  (venv) $ sudo apt install plantuml # Linux

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

.. note:: Our platform also includes integration tests that asses the overall performance of our framework using the default settings on a few standard ML datasets. By default these tests are not executed, to run them, set an environmental variable called `FORESHADOW_TESTS` to `ALL` 

Suggested development work flow
   1. Create a branch off of development to contain your change
   
      .. code-block:: console
      
         (venv) $ git checkout development
         (venv) $ git checkout -b {your_feature}

   2. Run pytest and pre-commit while developing
      This will help ensure something hasn't broken while adding a feature. Pre-commit will lint the code before each commit.
   
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
   
      Now you can go to the project on your github page and submit a pull request to the main project.

      .. note::  Make sure to submit the pull request against the development branch.


Adding Transformers
-------------------

Adding transformers is quite simple. Simply write a class with the `fit` `transform` and `inverse_transform` methods that extends :py:class:`scikit_learn.base.BaseEstimator` and  :py:class:`sklearn.base.TransformerMixin`. Take a look at the structure below and modify it to suit your needs. We would recommend taking a look at the `sklearn.preprocessing.RobustScaler`_ source code for a good example.

.. _sklearn.preprocessing.RobustScaler: https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/preprocessing/data.py#L939

.. code-block:: python

   from foreshadow.base import TransformerMixin, BaseEstimator
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

Future Architecture Roadmap
---------------------------

In progress
