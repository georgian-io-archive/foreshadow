# flake8: noqa
# isort: noqa
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR


class EstimatorFamilyMixin:
    def get_estimator_by_type(self, problem_type):
        return self.estimators[problem_type]


class LinearEstimatorFamily(EstimatorFamilyMixin):
    def __init__(self):
        self.estimators = {
            "classification": LogisticRegression(),
            "regression": LinearRegression(),
        }


class SVMEstimatorFamily(EstimatorFamilyMixin):
    def __init__(self):
        self.estimators = {
            "classification": LinearSVC(),
            "regression": LinearSVR(),
        }


class RFEstimatorFamily(EstimatorFamilyMixin):
    def __init__(self):
        self.estimators = {
            "classification": RandomForestClassifier(),
            "regression": RandomForestRegressor(),
        }


class NNEstimatorFamily(EstimatorFamilyMixin):
    def __init__(self):
        self.estimators = {
            "classification": MLPClassifier(),
            "regression": MLPRegressor(),
        }


class EstimatorFactory:
    def __init__(self):
        self.registered_estimator_families = {
            "Linear": LinearEstimatorFamily(),
            "SVM": SVMEstimatorFamily(),
            "RF": RFEstimatorFamily(),
            "NN": NNEstimatorFamily(),
        }

    def get_estimator(self, family, problem_type):
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
