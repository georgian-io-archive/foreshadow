"""Estimator factory and supported estimators."""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR


class EstimatorFamilyMixin:
    """Mixin class that allows an instance to retrieve an estimator.

    Retrieve is based on the type of the problem (classification or
    regression).
    """

    def get_estimator_by_type(self, problem_type):
        """Retrieve an estimator based on the problem type.

        Args:
            problem_type: either classification or regression

        Returns:
            an estimator of a particular model family

        """
        return self.estimators[problem_type]


class LinearEstimatorFamily(EstimatorFamilyMixin):
    """Linear model estimator family.

    Estimators include LinearRegression and LogisticRegression.
    """

    def __init__(self):
        self.estimators = {
            "classification": LogisticRegression(),
            "regression": LinearRegression(),
        }


class SVMEstimatorFamily(EstimatorFamilyMixin):
    """SVM model estimator family.

    Estimators include LinearSVC and LinearSVR.
    """

    def __init__(self):
        self.estimators = {
            "classification": LinearSVC(),
            "regression": LinearSVR(),
        }


class RFEstimatorFamily(EstimatorFamilyMixin):
    """RandomForest model estimator family.

    Estimators include RandomForestClassifier and RandomForestRegressor.
    """

    def __init__(self):
        self.estimators = {
            "classification": RandomForestClassifier(),
            "regression": RandomForestRegressor(),
        }


class NNEstimatorFamily(EstimatorFamilyMixin):
    """Neural Network model estimator family.

    Estimators include MLPClassifier and MLPRegressor
    """

    def __init__(self):
        self.estimators = {
            "classification": MLPClassifier(),
            "regression": MLPRegressor(),
        }


class EstimatorFactory:
    """Factory class that retrieves an estimator.

    Retrieval is based on the model family and the problem type.

    TODO we can add a register method if we decide to add more model
    algorithm family here.
    """

    def __init__(self):
        self.registered_estimator_families = {
            "Linear": LinearEstimatorFamily(),
            "SVM": SVMEstimatorFamily(),
            "RF": RFEstimatorFamily(),
            "NN": NNEstimatorFamily(),
        }

    def get_estimator(self, family, problem_type):
        """Retrieve the estimator based on the model family and problem type.

        Args:
            family: model family type
            problem_type: either classification or regression

        Returns:
            an estimator

        Raises:
            KeyError: unknow family type or problem type

        """
        if family not in self.registered_estimator_families:
            raise KeyError("Unknown model family type: {}".format(family))
        if problem_type not in ["classification", "regression"]:
            raise KeyError(
                "Only classification and regression are "
                "supported, unknown operation type: {}".format(problem_type)
            )
        return self.registered_estimator_families[
            family
        ].get_estimator_by_type(problem_type)
