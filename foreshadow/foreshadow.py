"""Core end-to-end pipeline, foreshadow."""

import inspect
import warnings
from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from foreshadow.estimators.auto import AutoEstimator
from foreshadow.estimators.meta import MetaEstimator
from foreshadow.optimizers.param_mapping import _param_mapping
from foreshadow.preprocessor import Preprocessor
from foreshadow.utils import check_df


class Foreshadow(BaseEstimator):
    """An end-to-end pipeline to preprocess and tune a machine learning model.

    Example:
        >>> shadow = Foreshadow()

    Args:
        X_preprocessor \
            (:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`, \
            optional): Preprocessor instance that will apply to X data. Passing
            False prevents the automatic generation of an instance.
        y_preprocessor \
            (:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`, \
            optional): Preprocessor instance that will apply to y data. Passing
            False prevents the automatic generation of an instance.
        estimator (:obj:`sklearn.base.BaseEstimator`, optional): Estimator
            instance to fit on processed data
        optimizer (:class:`sklearn.grid_search.BaseSeachCV`, optional):
            Optimizer class to optimize feature engineering and model
            hyperparameters

    """

    def __init__(
        self,
        X_preprocessor=None,
        y_preprocessor=None,
        estimator=None,
        optimizer=None,
    ):
        self.X_preprocessor = X_preprocessor
        self.y_preprocessor = y_preprocessor
        self.estimator = estimator
        self.optimizer = optimizer
        self.pipeline = None
        self.data_columns = None

        if isinstance(self.estimator, AutoEstimator) and optimizer is not None:
            warnings.warn(
                "An automatic estimator cannot be used with an optimizer."
                " Proceeding without use of optimizer"
            )
            self.optimizer = None

    @property
    def X_preprocessor(self):
        """Preprocessor object for performing feature engineering on X data.

        :getter: Returns Preprocessor object

        :setter: Verifies Preprocessor object, if None, creates a default
            Preprocessor

        :type: :obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`

        .. # noqa: I201
        """
        return self._X_preprocessor

    @X_preprocessor.setter
    def X_preprocessor(self, dp):
        if dp is not None:
            if dp is False:
                self._X_preprocessor = None
            elif isinstance(dp, Preprocessor):
                self._X_preprocessor = dp
            else:
                raise ValueError("Invalid value passed as X_preprocessor")
        else:
            self._X_preprocessor = Preprocessor()

    @property
    def y_preprocessor(self):
        """Preprocessor object for performing scaling and encoding on Y data.

        :getter: Returns Preprocessor object

        :setter: Verifies Preprocessor object, if None, creates a default
            Preprocessor

        :type: :obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`

        .. # noqa: I201
        """
        return self._y_preprocessor

    @y_preprocessor.setter
    def y_preprocessor(self, yp):
        if yp is not None:
            if yp is False:
                self._y_preprocessor = None
            elif isinstance(yp, Preprocessor):
                self._y_preprocessor = yp
            else:
                raise ValueError("Invalid value passed as y_preprocessor")
        else:
            self._y_preprocessor = Preprocessor(y_var=True)

    @property
    def estimator(self):
        """Estimator object for fitting preprocessed data.

        :getter: Returns Estimator object

        :setter: Verifies Estimator object. If None, an
            :obj:`AutoEstimator <foreshadow.estimators.AutoEstimator>`
            object is created in place.

        :type: :obj:`sklearn.base.BaseEstimator`

        .. # noqa: I201
        """
        return self._estimator

    @estimator.setter
    def estimator(self, e):
        if e is not None:
            if isinstance(e, BaseEstimator):
                self._estimator = e
            else:
                raise ValueError("Invalid value passed as estimator")
        else:
            self._estimator = AutoEstimator(
                include_preprocessors=False
                if self.X_preprocessor is not None
                else True
            )

    @property
    def optimizer(self):
        """Optimizer class that will fit the model.

        Performs a grid or random search algorithm on the parameter space from
        the preprocessors and estimators in the pipeline

        :getter: Returns optimizer class

        :setter: Verifies Optimizer class, defaults to None

        .. # noqa: I201
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, o):
        if o is None or (inspect.isclass(o) and issubclass(o, BaseSearchCV)):
            self._optimizer = o
        else:
            raise ValueError("Invalid value passed as optimizer")

    def fit(self, data_df, y_df):
        """Fit the Foreshadow instance using the provided input data.

        Args:
            data_df (:obj:`DataFrame <pandas.DataFrame>`): The input feature(s)
            y_df (:obj:`DataFrame <pandas.DataFrame>`): The response feature(s)

        Returns:
            :obj:`Foreshadow`: The fitted instance.

        """
        X_df = check_df(data_df)
        y_df = check_df(y_df)
        self.data_columns = X_df.columns.values.tolist()

        # setup MetaEstimator if y_preprocessor is passed in
        if self.y_preprocessor is not None:
            self.estimator = MetaEstimator(self.estimator, self.y_preprocessor)

        if self.X_preprocessor is not None:
            self.pipeline = Pipeline(
                [
                    ("preprocessor", self.X_preprocessor),
                    ("estimator", self.estimator),
                ]
            )
        else:
            self.pipeline = Pipeline([("estimator", self.estimator)])

        if self.optimizer is not None:
            # Calculate parameter search space
            param_ranges = _param_mapping(deepcopy(self.pipeline), X_df, y_df)

            self.opt_instance = self.optimizer(self.pipeline, param_ranges)
            self.opt_instance.fit(X_df, y_df)
            self.pipeline = self.opt_instance.best_estimator_
            # extract trained preprocessors
            if self.X_preprocessor is not None:
                self.X_preprocessor = self.opt_instance.best_estimator_.steps[
                    0
                ][1]
            if self.y_preprocessor is not None:
                self.y_preprocessor = self.opt_instance.best_estimator_.steps[
                    1
                ][1].preprocessor
        else:
            self.pipeline.fit(X_df, y_df)

        return self

    def _prepare_predict(self, pred_cols):
        """Validate prior to predicting.

        Args:
            pred_cols (:obj:`Index pandas.Index`): the predicted columns

        Raises:
            ValueError: Pipeline not fit yet
            ValueError: Predict must have the same columns as train

        """
        if self.pipeline is None:
            raise ValueError("Foreshadow has not been fit yet")
        elif pred_cols.values.tolist() != self.data_columns:
            raise ValueError(
                "Predict must have the same columns as train columns"
            )

    def predict(self, data_df):
        """Use the trained estimator to predict the response variable.

        Args:
            data_df (:obj:`DataFrame <pandas.DataFrame>`): The input feature(s)

        Returns:
            :obj:`DataFrame <pandas.DataFrame>`: The response feature(s) \
                (transformed if necessary)

        """
        data_df = check_df(data_df)
        self._prepare_predict(data_df.columns)
        return self.pipeline.predict(data_df)

    def predict_proba(self, data_df):
        """Use the trained estimator to predict the response variable.

        Uses the predicted confidences instead of binary predictions.

        Args:
            data_df (:obj:`DataFrame <pandas.DataFrame>`): The input feature(s)

        Returns:
            :obj:`DataFrame <pandas.DataFrame>`: The probability associated \
                with each response feature

        """
        data_df = check_df(data_df)
        self._prepare_predict(data_df.columns)
        return self.pipeline.predict_proba(data_df)

    def score(self, data_df, y_df=None, sample_weight=None):
        """Use the trained estimator to compute the evaluation score.

        The scoding method is defined by the selected estimator.

        Args:
            data_df (:obj:`DataFrame <pandas.DataFrame>`): The input feature(s)
            y_df (:obj:`DataFrame <pandas.DataFrame>`, optional): The response
                feature(s)
            sample_weight (:obj:`numpy.ndarray`, optional): The weights to be
                used when scoring each sample

        Returns:
            float: A computed prediction fitness score

        """
        data_df = check_df(data_df)
        y_df = check_df(y_df)
        self._prepare_predict(data_df.columns)
        return self.pipeline.score(data_df, y_df, sample_weight)
