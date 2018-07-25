"""
Main foreshadow object
"""

import operator
import warnings

from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from .preprocessor import Preprocessor
from .estimators.auto_estimator import AutoEstimator
from .estimators.meta_estimator import MetaEstimator
from .utils import check_df
from .optimizers.param_mapping import param_mapping


class Foreshadow(BaseEstimator):
    """Scikit-learn pipeline wrapper that preprocesses and automatically tunes
    a machine learning model
    
    Args:
        X_preprocessor (optional): An instance of foreshadow.Preprocessor
        y_preprocessor (optional): An instance of foreshadow.Preprocessor
        estimator (optional): An instance of an estimator that extends 
                              sklearn.BaseEstimator
        optimizer (optional): An instance of an optimizer that extends 
                              sklearn.BaseSearchCV
    """

    def __init__(
        self, X_preprocessor=None, y_preprocessor=None, estimator=None, optimizer=None
    ):
        self.X_preprocessor = X_preprocessor
        self.y_preprocessor = y_preprocessor
        self.estimator = estimator  # check status of X_preprocessor
        self.optimizer = optimizer
        self.target = None
        self.pipeline = None
        self.data_columns = None

        if isinstance(self.estimator, AutoEstimator) and not optimizer is None:
            warnings.warn(
                "An automatic estimator cannot be used with an optimizer."
                " Proceeding without use of optimizer"
            )
            self.optimizer = None

    X_preprocessor = property(operator.attrgetter("_X_preprocessor"))

    @X_preprocessor.setter
    def X_preprocessor(self, dp):
        if dp is not None:
            if dp == False:
                self._X_preprocessor = None
            elif isinstance(dp, Preprocessor):
                self._X_preprocessor = dp
            else:
                raise ValueError("Invalid value passed as X_preprocessor")
        else:
            self._X_preprocessor = Preprocessor()

    y_preprocessor = property(operator.attrgetter("_y_preprocessor"))

    @y_preprocessor.setter
    def y_preprocessor(self, yp):
        if yp is not None:
            if yp == False:
                self._y_preprocessor = None
            elif isinstance(yp, Preprocessor):
                self._y_preprocessor = yp
            else:
                raise ValueError("Invalid value passed as y_preprocessor")
        else:
            self._y_preprocessor = Preprocessor()

    estimator = property(operator.attrgetter("_estimator"))

    @estimator.setter
    def estimator(self, e):
        if e is not None:
            if isinstance(e, BaseEstimator):
                self._estimator = e
            else:
                raise ValueError("Invalid value passed as estimator")
        else:
            self._estimator = AutoEstimator(
                include_preprocessors=False if self.X_preprocessor is not None else True
            )

    optimizer = property(operator.attrgetter("_optimizer"))

    @optimizer.setter
    def optimizer(self, o):
        if o is not None:
            if isinstance(o, BaseSearchCV):
                self._optimizer = o
            else:
                raise ValueError("Invalid value passed as optimizer")
        else:
            self._optimizer = o

    def fit(self, data_df, y_df):
        """Fits the Foreshadow instance using the provided input data

        Args:
            data_df (pandas.DataFrame or numpy.ndarray or list): The input feature(s)
            y_df (pandas.DataFrame or numpy.ndarray or list): The response feature(s)
        """
        X_df = check_df(data_df)
        y_df = check_df(y_df)
        self.data_columns = X_df.columns.values.tolist()

        # setup MetaEstimator if y_preprocessor is passed in
        if self.y_preprocessor is not None:
            self.estimator = MetaEstimator(self.estimator, self.y_preprocessor)

        if self.X_preprocessor is not None:
            self.pipeline = self.X_preprocessor.pipeline
            self.pipeline.steps.append(["estimator", self.estimator])
        else:
            self.pipeline = Pipeline([("estimator", self.estimator)])

        if self.optimizer is not None:
            param_ranges = param_mapping(self.pipeline.get_params())
            self.optimizer(self.pipeline, param_ranges)
        else:
            self.pipeline.fit(X_df, y_df)

        return self

    def _prepare_predict(self, pred_cols):
        """Validates prior to predicting"""
        if self.pipeline is None:
            raise ValueError("Foreshadow has not been fit yet")
        elif pred_cols.values.tolist() != self.data_columns:
            raise ValueError("Predict must have the same columns as train columns")

    def predict(self, data_df):
        """Uses the trained estimator to predict the response for an input dataset

        Args:
            data_df (pandas.DataFrame or numpy.ndarray or list): The input feature(s)

        Returns:
            pandas.DataFrame: The response feature(s) (transformed if necessary)
        """
        data_df = check_df(data_df)
        self._prepare_predict(data_df.columns)
        return self.pipeline.predict(data_df)

    def predict_proba(self, data_df):
        """Uses the trained estimator to predict the probabilities of responses
        for an input dataset

        Args:
            data_df (pandas.DataFrame or numpy.ndarray or list): The input feature(s)

        Returns:
            pandas.DataFrame: The probability associated with each response feature
        """
        data_df = check_df(data_df)
        self._prepare_predict(data_df.columns)
        return self.pipeline.predict_proba(data_df)

    def score(self, data_df, y_df=None, sample_weight=None):
        """Uses the trained estimator to compute the evaluation score defined
        by the estimator

        Args:
            data_df (pandas.DataFrame or numpy.ndarray or list): The input feature(s)
            y_df (pandas.DataFrame or numpy.ndarray or list): The response feature(s)
            sample_weight (numpy.ndarray, optional): The weights to be used when scoring
                each sample

        Returns:
            (float): A computed prediction fitness score
        """
        data_df = check_df(data_df)
        y_df = check_df(y_df)
        self._prepare_predict(data_df.columns)
        return self.pipeline.score(data_df, y_df, sample_weight)
