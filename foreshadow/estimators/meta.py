"""
Wrapped Estimator
"""

from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import check_df


class MetaEstimator(BaseEstimator):
    """A wrapper for estimators that allows data preprocessing on the response 
    variable(s) using Preprocessor
    
    Args:
        estimator: An instance of a subclass of sklearn.BaseEstimator
        preprocessor: An instance of foreshadow.Preprocessor
    """

    def __init__(self, estimator, preprocessor):
        self.estimator = estimator
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        """Fits the AutoEstimator instance using a selected automatic machine learning
        estimator

        Args:
            X (pandas.DataFrame or numpy.ndarray or list): The input feature(s)
            y (pandas.DataFrame or numpy.ndarray or list): The response feature(s)
        """
        X = check_df(X)
        y = check_df(y)
        y = self.preprocessor.fit_transform(y)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Uses the trained estimator to predict the response for an input dataset

        Args:
            X (pandas.DataFrame or numpy.ndarray or list): The input feature(s)

        Returns:
            pandas.DataFrame: The response feature(s) (transformed)
        """
        X = check_df(X)
        return self.preprocessor.inverse_transform(self.estimator.predict(X))

    def predict_proba(self, X):
        """Uses the trained estimator to predict the probabilities of responses
        for an input dataset

        Args:
            X (pandas.DataFrame or numpy.ndarray or list): The input feature(s)

        Returns:
            pandas.DataFrame: The probability associated with each response feature
        """
        X = check_df(X)
        return self.estimator.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        """Uses the trained estimator to compute the evaluation score defined
        by the estimator

        Args:
            X (pandas.DataFrame or numpy.ndarray or list): The input feature(s)
            y (pandas.DataFrame or numpy.ndarray or list): The response feature(s)
            sample_weight (numpy.ndarray, optional): The weights to be used when scoring
                each sample
        
        Returns:
            float: A computed prediction fitness score
        """
        X = check_df(X)
        y = check_df(y)
        y = self.preprocessor.transform(y)
        return self.estimator.score(X, y, sample_weight)
