"""Classes that hold constants in foreshadow."""


class ProblemType:
    """Constants for problem types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class EstimatorFamily:
    """Constants for estimator families."""

    LINEAR = "linear"
    SVM = "svm"
    RF = "random_forest"
    NN = "neural_network"