"""
Definitions for classes related to n-gram featurization.

This includes class definitions for:
    -- NGramEmbedder
    -- NGramFeaturizer
    -- NGramFeaturizerBuilder
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .base_embedder import BaseEmbedder
from .base_text_featurizer import BaseTextFeaturizer


class NGramEmbedder(BaseEmbedder):
    """
    Concrete class to wrap sklearn's CountVectorizer Implementation.

    Is a component to NGramFeaturizer.

    Attributes:
        embedder
            -- sklearn's CountVectorizer
        ngram_range {Tuple[int, int]}
            -- Lower and upper bound of character tokens to use in featurization,
               inclusive. For example, (2, 3) correspond to character-level
               bigrams and trigrams.
        cutoff {Union[int, float]}
            -- Threshold that has to be exceeded in order to include ngram.
               If a float is provided, threshold is count relative to the dataset.
               If an int is provided, threshold is count.

    Method:
        load -- Load the embedder
        fit -- Fit the embedder to the training data
        transform -- Transfrom text to embeddings
    """

    def __init__(
        self, ngram_range: Tuple[int, int], cutoff: Union[int, float]
    ):
        """
        Init function.

        Arguments:
            ngram_range {Tuple[int, int]}
                -- Lower and upper bound of character tokens to use in featurization,
                   inclusive. For example, (2, 3) correspond to character-level
                   bigrams and trigrams.
            cutoff {Union[int, float]}
                -- Threshold that has to be exceeded in order to include ngram.
                   If a float is provided, threshold is count relative to the dataset.
                   If an int is provided, threshold is count.
        """
        self.ngram_range = ngram_range
        self.cutoff = cutoff
        self.embedder = self.load()

    def load(self) -> CountVectorizer:
        """Load an sklearn CountVectorizer instance."""
        return CountVectorizer(
            analyzer="char",
            strip_accents="unicode",
            ngram_range=self.ngram_range,
            min_df=self.cutoff,
        )

    def fit(self, vals: List[str]):
        """Fit the embedder to `vals`."""
        return self.embedder.fit(vals)

    def transform(self, vals: List[str]) -> pd.DataFrame:
        """Transform a list of words into embeddings."""
        return pd.DataFrame(self.embedder.transform(vals).toarray())

    def serialize(self) -> dict:
        """
        Return a serializable represenation of an NGramEmbedder.

        Raises:
            AssertionError -- Embedder is not yet fitted.

        Returns:
            dict -- Object attributes.
        """
        if not hasattr(self.embedder, "vocabulary_"):
            raise AssertionError("Exporting an unfitted CountVectorizer.")
        return vars(self)

    @staticmethod
    def deserialize(serialization: dict) -> "NGramEmbedder":
        """
        Instantiate an NGramEmbedder from a serialization.

        Creates an NGramEmbedder like normal before updating the embedder
        attribute with a trained, serialized embedder.
        """
        instance = NGramEmbedder(
            ngram_range=serialization["ngram_range"],
            cutoff=serialization["cutoff"],
        )
        instance.embedder = serialization["embedder"]
        return instance


class NGramFeaturizer(BaseTextFeaturizer):
    """
    Concrete class to featurize text using n-grams.

    Is a component of DataSetParser concrete subclasses.

    Attributes:
        target_text {str}
            -- Metafeature columns to perform text featurization on.
               Valid values are {'attr', 'all'}.
               'all' corresponds to the attribute column and sample columns.
        normalizable {bool}
            -- Whether the generated feature should be normalized. (default: {False})
        attribute_embedder {NGramEmbedder}
            -- Embedder for the metafeature attribute column
        samples_embedder {Optional[NGramEmbedder]}
            -- Optional embedder for the metafeature sample columns

        Refer to superclass for additional attributes.

    Methods:
        featurize -- Create secondary text metafeatures
    """

    def __init__(
        self,
        target_text: str,
        ngram_range: Tuple[int, int] = (2, 3),
        cutoff: Union[int, float] = 0.005,
        normalizable: bool = False,
    ):
        """
        Init function.

        Arguments:
            target_text {str}
                -- Metafeature columns to perform text featurization on.
                   Valid values are {'attr', 'all'}. 'all' corresponds to the
                   attribute column and sample columns.

        Keyword Arguments:
            ngram_range {Tuple[int, int]}
                -- Lower and upper bound of character tokens to use in featurization,
                   inclusive. For example, (2, 3) correspond to character-level
                   bigrams and trigrams.
            cutoff {Union[int, float]}
                -- Threshold that has to be exceeded in order to include ngram.
                   If a float is provided, threshold is count relative to the dataset.
                   If an int is provided, threshold is count.
            normalizable {bool}
                -- Whether the generated feature should be normalized. (default: {False})

        Raises:
            ValueError -- If invalid `target_text` is specified.
        """
        super().__init__(target_text=target_text, normalizable=normalizable)

        self.attribute_embedder = NGramEmbedder(
            ngram_range=ngram_range, cutoff=cutoff
        )
        if self.samples_column:
            self.samples_embedder = NGramEmbedder(
                ngram_range=ngram_range, cutoff=cutoff
            )

    def _update_feature_names(self):
        """
        Set `sec_feature_names` attribute with relevant feature names.

        Feature column corresponding to each column of secondary metafeature generated.
        Also marks whether a secondary metafeature column is not normalizable or
        not based on the `normalizable` attribute.
        """
        names = []
        attr_names = [
            f"ngrams_attr_{v}"
            for v in NGramFeaturizer.__sort_vocabs(
                self.attribute_embedder.embedder.vocabulary_
            )
        ]
        names += super()._mark_nonnormalizable(
            attr_names, normalizable=self.normalizable
        )

        if self.samples_embedder:
            samples_names = [
                f"ngrams_samples_{v}"
                for v in NGramFeaturizer.__sort_vocabs(
                    self.samples_embedder.embedder.vocabulary_
                )
            ]
            names += super()._mark_nonnormalizable(
                samples_names, normalizable=self.normalizable
            )

        self.sec_feature_names = names

    @staticmethod
    def __sort_vocabs(vocabs: Dict[str, int]) -> List[str]:
        return sorted(vocabs.keys(), key=lambda x: vocabs[x])

    def serialize(self) -> dict:
        """Return a serializable representation."""
        serialization = super().serialize()
        serialization["ngram_range"] = serialization[
            "attribute_embedder_config"
        ]["ngram_range"]
        serialization["cutoff"] = serialization["attribute_embedder_config"][
            "cutoff"
        ]
        return serialization


class NGramFeaturizerBuilder:
    """Builder class for NGramFeaturizer."""

    def __call__(
        self,
        target_text: str,
        ngram_range: Tuple[int, int],
        cutoff: Union[int, float],
        normalizable: bool,
        sec_feature_names: Optional[List[str]] = None,
        attribute_embedder_config: Optional[dict] = None,
        samples_embedder_config: Optional[dict] = None,
        **_ignore,
    ):
        """Build a NGramFeaturizer based on supplied keyword arguments."""
        featurizer = NGramFeaturizer(
            target_text=target_text,
            ngram_range=ngram_range,
            cutoff=cutoff,
            normalizable=normalizable,
        )

        featurizer.sec_feature_names = sec_feature_names
        if attribute_embedder_config is not None:
            featurizer.attribute_embedder = NGramEmbedder.deserialize(
                attribute_embedder_config
            )
        if samples_embedder_config is not None:
            featurizer.samples_embedder = NGramEmbedder.deserialize(
                samples_embedder_config
            )

        return featurizer
