"""AutoEstimator."""

import warnings

import numpy as np

from foreshadow.base import BaseEstimator
from foreshadow.estimators.config import get_tpot_config
from foreshadow.logging import logging
from foreshadow.utils import check_df, check_module_installed


class AutoEstimator(BaseEstimator):
    """A wrapped estimator that selects the solution for a given problem.

    By default each automatic machine learning solution runs for 1 minute but
    that can be changed through passed kwargs. Autosklearn is not required for
    this to work but if installed it can be used alongside TPOT.

    Args:
        problem_type (str): The problem type, 'regression' or 'classification'
        auto (str): The automatic estimator, 'tpot' or 'autosklearn'
        include_preprocessors (bool): Whether include preprocessors in AutoML
            pipelines
        estimator_kwargs (dict): A dictionary of args to pass to the specified
            auto estimator (both problem_type and auto must be specified)

    """

    def __init__(
        self,
        problem_type=None,
        auto=None,
        include_preprocessors=False,
        estimator_kwargs=None,
    ):
        self.problem_type = problem_type
        self.auto = auto
        self.include_preprocessors = include_preprocessors
        self.estimator_kwargs = estimator_kwargs
        self.estimator_class = None
        self.estimator = None

    @property
    def problem_type(self):
        """Type of machine learning problem.

        Either `regression` or `classification`.

        Returns:
            self._problem_type

        """
        return self._problem_type

    @problem_type.setter
    def problem_type(self, pt):
        """Set the problem type.

        Args:
            pt: problem type

        Raises:
            ValueError: pt not valid problem type

        """
        pt_options = ["classification", "regression"]
        if pt is not None and pt not in pt_options:
            raise ValueError("problem type must be in {}".format(pt_options))
        self._problem_type = pt

    @property
    def auto(self):
        """Type of automl package.

        Either `tpot` or `autosklearn`.

        Returns:
            self._auto, the type of automl package

        """
        return self._auto

    @auto.setter
    def auto(self, ae):
        """Set type of automl package.

        Args:
            ae: automl packaage

        Raises:
            ValueError: ae not a valid ae

        """
        ae_options = ["tpot", "autosklearn"]
        if ae is not None and ae not in ae_options:
            raise ValueError("auto must be in {}".format(ae_options))
        self._auto = ae

    @property
    def estimator_kwargs(self):
        """Get dictionary of kwargs to pass to AutoML package.

        Returns:
            estimator kwargs

        """
        return self._estimator_kwargs

    @estimator_kwargs.setter
    def estimator_kwargs(self, ek):
        """Set estimator kwargs.

        Args:
            ek: kwargs

        Raises:
            ValueError: ek not a valid kwargs dict.

        """
        if ek is not None and ek is not {}:
            if not isinstance(ek, dict) or not all(
                isinstance(k, str) for k in ek.keys()
            ):
                raise ValueError(
                    "estimator_kwargs must be a valid kwarg dictionary"
                )

            self._estimator_kwargs = ek
        else:
            self._estimator_kwargs = {}

    def _get_optimal_estimator_class(self):
        """Pick the optimal estimator class.

        Defaults to a working estimator if autosklearn is not installed.

        Returns:
            optimal estimator class.

        """
        auto_ = self._pick_estimator() if self.auto is None else self.auto

        estimator_choices = {
            "autosklearn": {
                "classification": "AutoSklearnClassifier",
                "regression": "AutoSklearnRegressor",
            },
            "tpot": {
                "classification": "TPOTClassifier",
                "regression": "TPOTRegressor",
            },
        }

        if not check_module_installed(auto_):
            selected_auto = "tpot"
            warnings.warn(
                "{} is not available, defaulting to {}".format(
                    auto_, selected_auto
                )
            )
            self.auto = selected_auto
        else:
            self.auto = auto_

        if self.auto == "autosklearn":
            import autosklearn.classification
            import autosklearn.regression

            class_ = estimator_choices[self.auto][self.problem_type]
            if class_ == "AutoSklearnClassifier":
                return getattr(autosklearn.classification, class_)
            else:
                return getattr(autosklearn.regression, class_)
        else:
            import tpot

            return getattr(
                tpot, estimator_choices[self.auto][self.problem_type]
            )

    def _pick_estimator(self):
        """Pick auto estimator based on benchmarked results.

        Returns:
            estimator

        """
        return "tpot" if self.problem_type == "regression" else "autosklearn"

    def _pre_configure_estimator_kwargs(self):
        """Configure auto estimators to perform similarly (time scale).

        Also remove preprocessors if necessary.

        Returns:
            estimator kwargs

        """
        if self.auto == "tpot" and "config_dict" not in self.estimator_kwargs:
            self.estimator_kwargs["config_dict"] = get_tpot_config(
                self.problem_type, self.include_preprocessors
            )
            if "max_time_mins" not in self.estimator_kwargs:
                self.estimator_kwargs["max_time_mins"] = 5
            if "verbosity" not in self.estimator_kwargs:
                self.estimator_kwargs["verbosity"] = 3
        elif (
            self.auto == "autosklearn"
            and not any(
                k in self.estimator_kwargs
                for k in ["include_preprocessors", "exclude_preprocessors"]
            )
            and not self.include_preprocessors
        ):
            self.estimator_kwargs["include_preprocessors"] = [
                "no_preprocessing"
            ]

        if (
            self.auto == "autosklearn"
            and "time_left_for_this_task" not in self.estimator_kwargs
        ):
            self.estimator_kwargs["time_left_for_this_task"] = 360

        return self.estimator_kwargs

    def construct_estimator(self, y):
        """Construct and return the auto estimator instance.

        Args:
            y: input labels

        Returns:
            autoestimator instance

        """
        self.problem_type = (
            determine_problem_type(y)
            if self.problem_type is None
            else self.problem_type
        )
        self.estimator_class = (
            self._get_optimal_estimator_class()
        )  # update estimator class in case of autodetect
        self._pre_configure_estimator_kwargs()  # validate estimator kwargs
        return self.estimator_class(**self.estimator_kwargs)

    def fit(self, X, y):
        """Fit the AutoEstimator instance.

        Uses the selected AutoML estimator.

        Args:
            X (pandas.DataFrame or numpy.ndarray or list): The input
                feature(s)
            y (pandas.DataFrame or numpy.ndarray or list): The response
                feature(s)

        Returns:
            The selected estimator

        """
        X = check_df(X)
        y = check_df(y)
        self._fit(X, y)

        return self.estimator

    def _fit(self, X, y):
        try:
            self.estimator = self.construct_estimator(y)
            self.estimator.fit(X, y)
        except RuntimeError as re:
            # if "a regression problem was provided to the TPOTClassifier " \
            #    "object" in str(re):
            logging.warning(
                "An error occurred from TPOT: {} Fall back "
                "to TPOT light option and retrain the "
                "model.".format(str(re))
            )
            self.estimator = self.construct_estimator(y)
            self.estimator.config_dict = "TPOT light"
            self.estimator.fit(X, y)

    def predict(self, X):
        """Use the trained estimator to predict the response.

        Args:
            X (pandas.DataFrame or numpy.ndarray or list): The input
                feature(s)

        Returns:
            pandas.DataFrame: The response feature(s)

        """
        X = check_df(X)
        return self.estimator.predict(X)

    def predict_proba(self, X):  # pragma: no cover
        """Use the trained estimator to predict the responses probabilities.

        Args:
            X (pandas.DataFrame or numpy.ndarray or list): The input
                feature(s)

        Returns:
            pandas.DataFrame: The probability associated with each response \
                feature

        """
        X = check_df(X)
        return self.estimator.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        """Use the trained estimator to compute the evaluation score.

        Note: sample weights are not supported

        Args:
            X (pandas.DataFrame or numpy.ndarray or list): The input feature(s)
            y (pandas.DataFrame or numpy.ndarray or list): The response
                feature(s)
            sample_weight: sample weighting. Not implemented.

        Returns:
            float: A computed prediction fitness score

        """
        X = check_df(X)
        y = check_df(y)
        return self.estimator.score(X, y)


def determine_problem_type(y):
    """Determine modeling problem type.

    Args:
        y: input labels

    Returns:
        problem type inferred from y.

    """
    return (
        "classification"
        if np.unique(y.values.ravel()).size == 2
        else "regression"
    )
