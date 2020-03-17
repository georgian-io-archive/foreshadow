"""BaseTextFeaturizer class definitions."""

import copy
from abc import abstractmethod
from typing import List, Optional, Sequence

import pandas as pd

from .base_featurizer import BaseFeaturizer


class BaseTextFeaturizer(BaseFeaturizer):
    """
    Abstract class to extract secondary metafeatures from text in (test) metafeatures.

    Attributes:
        target_text {str}
            -- Metafeature columns to perform text featurization on.
               Valid values are {'attr', 'all'}. 'all' corresponds to the
               attribute column and sample columns.
        n_samples_cols {int}
            -- Number of sample columns in (test) metafeatures dataframe. (default: {5})
        normalizable {bool}
            -- Whether the generated feature should be normalized. (default: {False})
        attribute_column {str}
            -- Text column corresponding to the attribute column of (test)
               metafeatures.
        samples_column {str}
            -- Text column corresponding to the attribute column of (test)
               metafeatures.
        attribute_embedder
            -- Text embedder for the attribute column.
        samples_embedder
            -- Text embedder for the sample columns.
    """

    def __init__(
        self,
        target_text: str,
        n_sample_cols: int = 5,
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
            n_sample_cols {int}
                -- Number of sample columns in (test) metafeatures dataframe.
                   (default: {5})
            normalizable {bool}
                -- Whether the generated feature should be normalized.
                   (default: {False})

        Raises:
            ValueError -- If invalid value of `target_text` is provided.
        """
        super().__init__(method=None, normalizable=normalizable)

        self.target_text = target_text
        if self.target_text not in {"attr", "all"}:
            raise ValueError(
                "Invalid `target_text` provided. "
                "Valid values are {'attr', 'all'}."
            )

        self.n_sample_cols = n_sample_cols

        self.attribute_column: Sequence[str] = None
        self.samples_column: Sequence[str] = None

        self.attribute_embedder = None
        self.samples_embedder = None

        if self.target_text == "attr":
            self.attribute_column = ["attribute_name"]
        elif self.target_text == "all":
            self.attribute_column = ["attribute_name"]
            self.samples_column = [
                f"sample{i + 1}" for i in range(self.n_sample_cols)
            ]

    def featurize(
        self,
        meta_df: Optional[pd.DataFrame] = None,
        test_meta_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Featurize text by converting them into embeddings.

        Fit an attribute (and a separate samples) vectorizer(s) to text
        features before converting them into embeddings.

        At least one keyword argument must be provided.

        Keyword Arguments:
            meta_df {pd.DataFrame} -- Training metafeatures. (default: {None})
            test_meta_df {Optional[pd.DataFrame]} -- Test metafeatures.
                                                     (default: {None})
        """
        if meta_df is None and test_meta_df is None:
            raise ValueError("At least one keyword argument must be provided.")

        if meta_df is not None:
            if self.samples_column:
                # Train embedders
                attr_texts = BaseTextFeaturizer.preprocess_texts(
                    meta_df, self.attribute_column
                )
                sample_texts = BaseTextFeaturizer.preprocess_texts(
                    meta_df, self.samples_column
                )
                self.attribute_embedder.fit(attr_texts)
                self.samples_embedder.fit(sample_texts)

                # Transform texts for training data
                self.sec_metafeatures = pd.concat(
                    [
                        self.attribute_embedder.transform(attr_texts),
                        self.samples_embedder.transform(sample_texts),
                    ],
                    axis=1,
                )
            else:
                # Same as above, but only for attribute texts
                attr_texts = BaseTextFeaturizer.preprocess_texts(
                    meta_df, self.attribute_column
                )
                self.attribute_embedder.fit(attr_texts)
                self.sec_metafeatures = self.attribute_embedder.transform(
                    attr_texts
                )

        if test_meta_df is not None:
            test_attr_texts = BaseTextFeaturizer.preprocess_texts(
                test_meta_df, self.attribute_column
            )
            if self.samples_column:
                test_samples_text = BaseTextFeaturizer.preprocess_texts(
                    test_meta_df, self.samples_column
                )

                self.sec_test_metafeatures = pd.concat(
                    [
                        self.attribute_embedder.transform(test_attr_texts),
                        self.samples_embedder.transform(test_samples_text),
                    ],
                    axis=1,
                )
            else:
                self.sec_test_metafeatures = self.attribute_embedder.transform(
                    test_attr_texts
                )

        # Update feature names now that their sizes are determined
        self._update_feature_names()

    def serialize(self) -> dict:
        """Return a serializable representation."""
        serialization = copy.deepcopy(vars(self))

        # Remove superfluous attributes for memory efficiency
        del serialization["sec_metafeatures"]
        del serialization["sec_test_metafeatures"]
        del serialization["attribute_embedder"]
        del serialization["samples_embedder"]

        # Serialize embedders
        if self.attribute_embedder is not None:
            serialization[
                "attribute_embedder_config"
            ] = self.attribute_embedder.serialize()
        if self.samples_embedder is not None:
            serialization[
                "samples_embedder_config"
            ] = self.samples_embedder.serialize()

        return serialization

    @abstractmethod
    def _update_feature_names(self):
        raise NotImplementedError

    @staticmethod
    def preprocess_texts(df: pd.DataFrame, cols: List[str]) -> List[str]:
        """
        Project a subset of `df` using `cols`.

        For each row in the projected dataframe, the strings are ':'-concatenated,
        as well as stripped and lowered.

        Arguments:
            df {pd.DataFrame} -- Dataframe containing text column(s).
            cols {List[str]} -- Columns in `df` to use for text preprocessing.

        Returns:
            List[str] -- A list whose members containing row-wise preprocessed texts.
        """
        return (
            df[cols]
            .apply(
                lambda row: ":".join(
                    str(s).strip().lower().replace("_", " ").replace("-", " ")
                    for s in row.values
                ),
                axis=1,
            )
            .values.ravel()
        )
