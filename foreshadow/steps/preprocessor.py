"""Defines the Preprocessor step in the Foreshadow DataPreparer pipeline."""
from sklearn.pipeline import make_pipeline

from foreshadow.config import config
from foreshadow.intents import Droppable, IntentType, Text
from foreshadow.smart import TextEncoder
from foreshadow.utils import AcceptedKey, DefaultConfig, Override

from .autointentmap import AutoIntentMixin
from .preparerstep import PreparerStep


def _configure_text_transformation_pipeline(num_of_non_text_features):
    n_components = (
        num_of_non_text_features
        if num_of_non_text_features > DefaultConfig.N_COMPONENTS_SVD
        else DefaultConfig.N_COMPONENTS_SVD
    )
    text_pipeline = make_pipeline(TextEncoder(n_components=n_components))
    return text_pipeline


class Preprocessor(PreparerStep, AutoIntentMixin):
    """Apply preprocessing steps to each column.

    Params:
        *args: args to PreparerStep constructor.
        **kwargs: kwargs to PreparerStep constructor.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline_by_intent = self._load_transformation_pipelines()

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
        res = super().transform(X=X)
        return res

    def _get_intent(self, column):
        override_key = "_".join([Override.INTENT, column])
        if (
            self.cache_manager.has_override()
            and override_key in self.cache_manager[AcceptedKey.OVERRIDE]
        ):
            intent = self.cache_manager[AcceptedKey.OVERRIDE][override_key]
        else:
            intent = self.cache_manager[AcceptedKey.INTENT][column]
        return intent

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
        # We need to handle the text data differently. Specifically, we will
        # group them together and process them with in pipeline instead of
        # one by one. In this pipeline, we want to apply a feature reducing
        # component, for now, a PCA as an extra step. This is to prevent the
        # TFIDF output to overwhelm the other columns. However, should we
        # encapsulate the modification here or in the TextEncoder? I believe
        # the latter makes more sense but we still need to separate the text
        # features from the rest.
        list_of_tuples = []
        text_features = []
        for column in X.columns:
            intent = self._get_intent(column=column)
            if intent == Text.__name__:
                text_features.append(column)
            elif intent == Droppable.__name__:
                continue
            else:
                transformation_pipeline = self.pipeline_by_intent[intent]
                list_of_tuples.append(
                    (column, transformation_pipeline, column)
                )
        if len(text_features) > 0:
            text_trans_pipeline = _configure_text_transformation_pipeline(
                num_of_non_text_features=len(X.columns) - len(text_features)
            )
            list_of_tuples.append(
                ("text_preprocessing", text_trans_pipeline, text_features)
            )
        return list_of_tuples
