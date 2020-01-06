"""Classes that hold constants in foreshadow."""


class DefaultConfig:
    """Constants for default configurations."""

    PROCESSED_DATA_EXPORT_PATH = "processed_data.csv"
    ENABLE_SAMPLING = True
    SAMPLING_DATASET_SIZE_THRESHOLD = 10000
    SAMPLING_WITH_REPLACEMENT = False
    SAMPLING_FRACTION = 0.2
    N_JOBS = 1


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


class ConfigKey:
    """Constants of configuration key in foreshadow."""

    SAMPLING_DATASET_SIZE_THRESHOLD = "sampling_dataset_size_threshold"
    ENABLE_SAMPLING = "enable_sampling"
    SAMPLING_WITH_REPLACEMENT = "with_replacement"
    SAMPLING_FRACTION = "sampling_fraction"
    N_JOBS = "n_jobs"
    PROCESSED_DATA_EXPORT_PATH = "processed_data_export_path"
    CUSTOMIZED_CLEANERS = "customized_cleaners"


class AcceptedKey:
    """Accepted keys of the CacheManager."""

    INTENT = "intent"
    DOMAIN = "domain"
    METASTAT = "metastat"
    GRAPH = "graph"
    OVERRIDE = "override"
    CONFIG = "config"
    CUSTOMIZED_TRANSFORMERS = "customized_transformers"


class Constant:
    """General constants in Foreshadow."""

    NAN_FILL_VALUE = "NaN"
