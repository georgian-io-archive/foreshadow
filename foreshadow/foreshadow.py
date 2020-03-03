"""Core end-to-end pipeline, foreshadow."""

import inspect
import warnings
from typing import List, NoReturn, Union

from sklearn.model_selection._search import BaseSearchCV

from foreshadow.base import BaseEstimator
from foreshadow.cachemanager import CacheManager
from foreshadow.concrete.internals.cleaners.customizable_base import (
    CustomizableBaseCleaner,
)
from foreshadow.estimators.auto import AutoEstimator
from foreshadow.estimators.estimator_wrapper import EstimatorWrapper
from foreshadow.intents import IntentType
from foreshadow.logging import logging
from foreshadow.pipeline import SerializablePipeline
from foreshadow.preparer import DataPreparer
from foreshadow.utils import (
    AcceptedKey,
    ConfigKey,
    Override,
    ProblemType,
    check_df,
)


class Foreshadow(BaseEstimator):
    """An end-to-end pipeline to preprocess and tune a machine learning model.

    Example:
        >>> shadow = Foreshadow(problem_type=ProblemType.CLASSIFICATION)

    Args:
        X_preparer \
            (:obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`, \
            optional): Preprocessor instance that will apply to X data. Passing
            False prevents the automatic generation of an instance.
        y_preparer \
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
        X_preparer=None,
        y_preparer=None,
        estimator=None,
        problem_type=None,
        optimizer=None,
        optimizer_kwargs=None,
    ):
        if problem_type not in [
            ProblemType.CLASSIFICATION,
            ProblemType.REGRESSION,
        ]:
            raise ValueError(
                "Unknown Problem Type {}. Please choose from {} "
                "or {}".format(
                    problem_type,
                    ProblemType.CLASSIFICATION,
                    ProblemType.REGRESSION,
                )
            )
        self.problem_type = problem_type
        self.X_preparer = X_preparer
        self.y_preparer = y_preparer
        self.estimator = estimator
        self.optimizer = optimizer
        self.optimizer_kwargs = (
            {} if optimizer_kwargs is None else optimizer_kwargs
        )
        self.pipeline = None
        self.data_columns = None
        self.has_fitted = False

        if isinstance(self.estimator, AutoEstimator) and optimizer is not None:
            warnings.warn(
                "An automatic estimator cannot be used with an optimizer."
                " Proceeding without use of optimizer"
            )
            self.optimizer = None

        if self.y_preparer is not None:
            self.estimator_wrapper = EstimatorWrapper(
                self.estimator, self.y_preparer
            )
        else:
            self.estimator_wrapper = self.estimator

    @property
    def X_preparer(self):  # noqa
        """Preprocessor object for performing feature engineering on X data.

        :getter: Returns Preprocessor object

        :setter: Verifies Preprocessor object, if None, creates a default
            Preprocessor

        :type: :obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`

        Returns:
            the X_preparer object

        .. # noqa: I201
        """
        return self._X_preprocessor

    @X_preparer.setter
    def X_preparer(self, dp):
        if dp is not None:
            if dp is False:
                self._X_preprocessor = None
            elif isinstance(dp, DataPreparer):
                self._X_preprocessor = dp
            else:
                raise ValueError(
                    "Invalid value: '{}' " "passed as X_preparer".format(dp)
                )
        else:
            self._X_preprocessor = DataPreparer(cache_manager=CacheManager())

    @property
    def y_preparer(self):  # noqa
        """Preprocessor object for performing scaling and encoding on Y data.

        :getter: Returns Preprocessor object

        :setter: Verifies Preprocessor object, if None, creates a default
            Preprocessor

        :type: :obj:`Preprocessor <foreshadow.preprocessor.Preprocessor>`

        Returns:
            the y_preparer object

        .. # noqa: I201
        """
        return self._y_preprocessor

    @y_preparer.setter
    def y_preparer(self, yp):
        if yp is not None:
            if yp is False:
                self._y_preprocessor = None
            elif isinstance(yp, DataPreparer):
                self._y_preprocessor = yp
            else:
                raise ValueError("Invalid value passed as y_preparer")
        else:
            self._y_preprocessor = DataPreparer(
                cache_manager=CacheManager(),
                y_var=True,
                problem_type=self.problem_type,
            )

    @property
    def estimator(self):  # noqa
        """Estimator object for fitting preprocessed data.

        :getter: Returns Estimator object

        :setter: Verifies Estimator object. If None, an
            :obj:`AutoEstimator <foreshadow.estimators.AutoEstimator>`
            object is created in place.

        :type: :obj:`sklearn.base.BaseEstimator`

        Returns:
            the estimator object

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
                if self.X_preparer is not None
                else True
            )

    @property
    def optimizer(self):  # noqa
        """Optimizer class that will fit the model.

        Performs a grid or random search algorithm on the parameter space from
        the preprocessors and estimators in the pipeline

        :getter: Returns optimizer class

        :setter: Verifies Optimizer class, defaults to None

        Returns:
            the optimizer object

        .. # noqa: I201
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, o):
        if o is None or (inspect.isclass(o) and issubclass(o, BaseSearchCV)):
            self._optimizer = o
        else:
            raise ValueError("Invalid optimizer: '{}' passed.".format(o))

    def _reset(self):
        if hasattr(self, "pipeline"):
            del self.pipeline
        if hasattr(self, "tuner"):
            del self.tuner
            del self.opt_instance

    def fit(self, data_df, y_df):
        """Fit the Foreshadow instance using the provided input data.

        Args:
            data_df (:obj:`DataFrame <pandas.DataFrame>`): The input feature(s)
            y_df (:obj:`DataFrame <pandas.DataFrame>`): The response feature(s)

        Returns:
            :obj:`Foreshadow`: The fitted instance.

        """
        self._reset()
        X_df = check_df(data_df)
        y_df = check_df(y_df)
        self.data_columns = X_df.columns.values.tolist()

        if self.X_preparer is not None:
            self.pipeline = SerializablePipeline(
                [
                    ("X_preparer", self.X_preparer),
                    ("estimator_wrapper", self.estimator_wrapper),
                ]
            )
        else:
            self.pipeline = SerializablePipeline(
                [("estimator_wrapper", self.estimator_wrapper)]
            )

        # TODO we may need this for future development but right now it's
        #  dragging down code coverage
        # if self.optimizer is not None:
        #     self.pipeline.fit(X_df, y_df)
        #     params = ParamSpec(self.pipeline, X_df, y_df)
        #     self.opt_instance = self.optimizer(
        #         estimator=self.pipeline,
        #         param_distributions=params,
        #         **{
        #             "iid": True,
        #             "scoring": "accuracy",
        #             "n_iter": 10,
        #             "return_train_score": True,
        #         }
        #     )
        #     self.tuner = Tuner(self.pipeline, params, self.opt_instance)
        #     self.tuner.fit(X_df, y_df)
        #     self.pipeline = self.tuner.transform(self.pipeline)
        #     # extract trained preprocessors
        #     if self.X_preparer is not None:
        #         self.X_preparer = self.pipeline.steps[0][1]
        #     if self.y_preparer is not None:
        #         self.y_preparer = self.opt_instance.best_estimator_.steps[1][
        #             1
        #         ].preprocessor
        # else:
        #     self.pipeline.fit(X_df, y_df)

        self.pipeline.fit(X_df, y_df)
        self.has_fitted = True

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

    # TODO all serialization/deserialization code has been turned off
    #  temporarily since we are not using it right now and it's dragging
    #  down the code coverage.
    # def dict_serialize(self, deep=False):
    #     """Serialize the init parameters of the foreshadow object.
    #
    #     Args:
    #         deep (bool): If True, will return the parameters for this
    #         estimator recursively
    #
    #     Returns:
    #         dict: The initialization parameters of the foreshadow object.
    #
    #     """
    #     serialized = super().dict_serialize(deep=False)
    #     serialized["estimator"] = self._customize_serialized_estimator(
    #         self.estimator
    #     )
    #     return serialized

    # @staticmethod
    # def _customize_serialized_estimator(estimator):
    #     if isinstance(estimator, AutoEstimator):
    #         """For third party automl estimator, the estimator_kwargs
    #         have different format and structure. To reduce verbosity,
    #         this field is removed from the serialized object.
    #         """
    #         serialized_estimator = estimator.serialize()
    #         serialized_estimator.pop("estimator_kwargs")
    #     else:
    #         serialized_estimator = estimator.get_params()
    #         serialized_estimator["_class"] = (
    #             estimator.__module__ + "." + type(estimator).__name__
    #         )
    #         serialized_estimator["_method"] = "dict"
    #
    #     result = serialized_estimator
    #     return result

    # @classmethod
    # def dict_deserialize(cls, data):
    #     """Deserialize the dictionary form of a foreshadow object.
    #
    #     Args:
    #         data: The dictionary to parse as foreshadow object is
    #         constructed.
    #
    #     Returns:
    #         object: A re-constructed foreshadow object.
    #
    #     """
    #     serialized_estimator = data.pop("estimator")
    #     estimator = cls._reconstruct_estimator(serialized_estimator)
    #
    #     params = _make_deserializable(data)
    #     data_columns = params.pop("data_columns")
    #     params["estimator"] = estimator
    #
    #     ret_tf = cls(**params)
    #     ret_tf.data_columns = data_columns
    #     return ret_tf
    #
    # @classmethod
    # def _reconstruct_estimator(cls, data):
    #     estimator_type = data.pop("_class")
    #     _ = data.pop("_method")
    #
    #     if estimator_type == AutoEstimator.__name__:
    #         class_name = estimator_type
    #         module_path = None
    #     else:
    #         class_name = estimator_type.split(".")[-1]
    #         module_path = ".".join(estimator_type.split(".")[0:-1])
    #
    #     estimator_class = get_transformer(class_name, source_lib=module_path)
    #     estimator = estimator_class()
    #     estimator.set_params(**data)
    #     return estimator

    def get_params(self, deep=True):
        """Get params for this object. See super.

        Args:
            deep: True to recursively call get_params, False to not.

        Returns:
            params for this object.

        """
        params = super().get_params(deep=deep)
        params["data_columns"] = self.data_columns
        return params

    def set_params(self, **params):
        """Set params for this object. See super.

        Args:
            **params: params to set.

        Returns:
            See super.

        """
        self.data_columns = params.pop("data_columns", None)
        return super().set_params(**params)

    def get_intent(self, column_name: str) -> Union[str, None]:
        """Retrieve the intent of a column.

        Args:
            column_name: the column name

        Returns:
            str: the intent of the column

        """
        # Note: this retrieves intent from cache_manager. Only columns have
        # been processed will be visible.
        cache_manager = self.X_preparer.cache_manager
        if self._has_column_in_cache_manager(column_name):
            return cache_manager[AcceptedKey.INTENT][column_name]
        else:
            logging.info(
                "No intent exists for column {}. Either the column "
                "doesn't exist or foreshadow object has not "
                "been fitted yet.".format(column_name)
            )
            return None

    def list_intent(self, column_names: List[str]) -> List[str]:
        """Retrieve the intent of a list of columns.

        Args:
            column_names: a list of columns

        Returns:
            The list of intents

        """
        return [self.get_intent(column) for column in column_names]

    def _has_column_in_cache_manager(self, column: str) -> Union[bool, None]:
        """Check if the column exists in the cache manager.

        If the foreshadow object has not been trained, it will return None.

        Args:
            column: the column name

        Returns:
            Whether a column exists in the cache manager

        """
        if not self.has_fitted:
            logging.info(
                "The foreshadow object is not trained yet. Please make sure "
                "the column {} exist to ensure the override takes "
                "effect.".format(column)
            )
            return False
        cache_manager = self.X_preparer.cache_manager
        return True if column in cache_manager[AcceptedKey.INTENT] else False

    def override_intent(self, column_name: str, intent: str) -> NoReturn:
        """Override the intent of a particular column.

        Args:
            column_name: the column to override
            intent: the user supplied intent

        Raises:
            ValueError: Invalid column to override.

        """
        if not IntentType.is_valid(intent):
            raise ValueError(
                "Invalid intent type {}. "
                "Supported intent types are {}.".format(
                    intent, IntentType.list_intents()
                )
            )

        if (
            not self._has_column_in_cache_manager(column_name)
            and self.has_fitted
        ):
            raise ValueError("Invalid Column {}".format(column_name))
        # Update the intent
        self.X_preparer.cache_manager[AcceptedKey.OVERRIDE][
            "_".join([Override.INTENT, column_name])
        ] = intent
        self.X_preparer.cache_manager[AcceptedKey.INTENT][column_name] = intent

    def configure_multiprocessing(self, n_job: int = 1) -> NoReturn:
        """Configure the multiprocessing option.

        Args:
            n_job: the number of processes to run the job.

        """
        self.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.N_JOBS
        ] = n_job

    def set_processed_data_export_path(
        self, data_path: str, is_train: bool
    ) -> NoReturn:
        """Set path to export data before feeding the data to the estimator.

        Args:
            data_path: the data path string
            is_train: whether this is for training data

        """
        key = (
            ConfigKey.PROCESSED_TRAINING_DATA_EXPORT_PATH
            if is_train
            else ConfigKey.PROCESSED_TEST_DATA_EXPORT_PATH
        )
        self.X_preparer.cache_manager[AcceptedKey.CONFIG][key] = data_path

    def pickle_fitted_pipeline(self, path: str) -> NoReturn:
        """Pickle the foreshadow object with the best pipeline estimator.

        Args:
            path: the pickle file path

        Raises:
            ValueError: pipeline not fitted.

        """
        if not self.has_fitted:
            logging.error("No pipeline has been fitted yet.")
            raise ValueError("The pipeline has not been fitted yet.")

        import pickle

        if (
            isinstance(self.estimator, AutoEstimator)
            and self.estimator.estimator.fitted_pipeline_ is not None
        ):
            self.estimator = self.estimator.estimator.fitted_pipeline_
            # updating the estimator above will not update the reference in
            # the pipeline instance as it still points to the old object.
            # Accessing the estimator wrapper object and update the
            # estimator field.
            self.pipeline.steps[1][1].estimator = self.estimator

        with open(path, "wb") as fopen:
            pickle.dump(self.pipeline, fopen)

    def configure_sampling(
        self,
        enable_sampling=True,
        sampling_fraction: float = 0.2,
        replace: bool = False,
    ) -> NoReturn:  # noqa: S001
        """Configure the sampling criteria.

        Args:
            enable_sampling: whether to enable sampling in data cleaning and intent resolving # noqa: E501
            sampling_fraction: whether to use replacement during sampling
            replace: the sampling fraction

        Returns:

        """
        self.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.ENABLE_SAMPLING
        ] = enable_sampling
        self.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.SAMPLING_FRACTION
        ] = sampling_fraction
        self.X_preparer.cache_manager[AcceptedKey.CONFIG][
            ConfigKey.SAMPLING_WITH_REPLACEMENT
        ] = replace

    def register_customized_data_cleaner(
        self, data_cleaners: List
    ) -> NoReturn:
        """**EXPERIMENTAL** Allow user to register a customized data cleaner.

        Args:
            data_cleaners: customized data cleaners

        Raises:
            ValueError: data cleaner must be a child class of the base cleaner.

        """
        for cleaner in data_cleaners:
            if not issubclass(cleaner, CustomizableBaseCleaner):
                raise ValueError(
                    "cleaner {} must be a child class of the {} class.".format(
                        str(cleaner), CustomizableBaseCleaner.__name__
                    )
                )

        self.X_preparer.cache_manager[AcceptedKey.CUSTOMIZED_TRANSFORMERS][
            ConfigKey.CUSTOMIZED_CLEANERS
        ] = data_cleaners
