"""Defines the Preprocessor step in the Foreshadow DataPreparer pipeline."""
from sklearn.pipeline import make_pipeline

from foreshadow.config import config
from foreshadow.intents import IntentType
from foreshadow.utils import AcceptedKey, Override

from .autointentmap import AutoIntentMixin
from .preparerstep import PreparerStep


class Preprocessor(PreparerStep, AutoIntentMixin):
    """Apply preprocessing steps to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformation_pipeline_by_intent = (
            self._load_transformation_pipelines()
        )

    def fit(self, X, *args, **kwargs):
        """Fit this step.

        calls underlying parallel process.

        Args:
            X: input DataFrame
            *args: args to _fit
            **kwargs: kwargs to _fit

        Returns:
            transformed data handled by Pipeline._fit

        """
        self.check_resolve(X)
        list_of_tuples = self._construct_column_transformer_tuples(X=X)
        self._prepare_feature_processor(list_of_tuples=list_of_tuples)
        self.feature_processor.fit(X=X)
        return self

    def transform(self, X, *args, **kwargs):
        """Clean the dataframe.

        Args:
            X: the data frame.
            *args: positional args.
            **kwargs: key word args.

        Returns:
            A transformed dataframe.

        """
        return super().transform(X=X)

    def _prepare_transformation_pipeline(self, intent, column):
        # TODO this must be some optimization we can do. Walk through both
        #  intent override cases again to check how it affects the logic.
        override_key = "_".join([Override.INTENT, column])
        if (
            self.cache_manager.has_override()
            and override_key in self.cache_manager[AcceptedKey.OVERRIDE]
        ):
            intent = self.cache_manager[AcceptedKey.OVERRIDE][override_key]
        return self.transformation_pipeline_by_intent[intent]

    def _load_transformation_pipelines(self):
        transformation_pipeline_by_intent = dict()
        for intent in IntentType.list_intents():
            transformers_class_list = config.get_preprocessor_steps(intent)
            if (
                transformers_class_list is not None
                and len(transformers_class_list) > 0
            ):
                transformer_list = [
                    tc(cache_manager=self.cache_manager)
                    # TODO: Allow kwargs in config
                    for tc in transformers_class_list
                ]
            else:
                transformer_list = ["passthrough"]

            transformation_pipeline = make_pipeline(*transformer_list)
            transformation_pipeline_by_intent[intent] = transformation_pipeline

        return transformation_pipeline_by_intent

    def _construct_column_transformer_tuples(self, X):
        list_of_tuples = []
        for column in X.columns:
            intent = self.cache_manager[AcceptedKey.INTENT, column]
            transformation_pipeline = self._prepare_transformation_pipeline(
                intent=intent, column=column
            )
            list_of_tuples.append((column, transformation_pipeline, column))
        return list_of_tuples
