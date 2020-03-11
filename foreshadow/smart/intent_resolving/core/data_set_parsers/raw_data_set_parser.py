"""Class definition for the RawDataSetParser abstract class."""

from pathlib import Path
from typing import Callable, Type

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .. import heuristics as hr
from .base_data_set_parser import DataSetParser


class RawDataSetParser(DataSetParser):
    """
    Abstract class to parse and extract metafeatures from a raw data set.

    Concrete classes that interface directly with raw data sets should inherit
    from here. Since subclasses from this class will interact with raw data sets
    solely for prediction, only the test-related attributes
    (test_metafeatures, test_labels) will be used.

    Attributes:
        Refer to DataSetParser superclass.

    Methods:
        normalize_features
            -- Performs normalization on the test metafeatures.
    """

    @staticmethod
    def _extract_base_features(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Extract base features from the data set.

        Base features include:
            {attribute_name, total_val,
             num_distincts, num_nans,
             max, min, mean, stddev}.

        Secondary feature `avg_val_len` is also included here to be consistent
        with other sibling class methods.

        Arguments:
            raw {pd.DataFrame} -- A cleaned dataframe from the raw CSV file.

        Returns:
            pd.DataFrame -- A dataframe containing the meta-features.
        """
        DEFAULT_PLACEHOLDER = 0.0
        COLUMNS = (
            "attribute_name",
            "total_val",
            "max",
            "min",
            "mean",
            "stddev",
            "num_nans",
            "num_distincts",
        )

        # Find raw feature columns that may be numeric
        may_be_numeric = raw.apply(hr.is_number_as_string, axis=0)

        def _descriptive_stat_reducer(
            reduce_fn: Callable
        ) -> Callable[[pd.Series], pd.Series]:
            return (
                lambda series: reduce_fn(hr.convert_to_numeric(series))
                if may_be_numeric[series.name]
                else DEFAULT_PLACEHOLDER
            )

        # Create base metafeatures as a data frame
        df = (
            pd.DataFrame(columns=COLUMNS, index=raw.columns)
            .pipe(
                lambda df: df.assign(
                    attribute_name=raw.columns,
                    total_val=[len(raw)] * len(raw.columns),
                    samples_set=raw.astype("object").apply(
                        lambda s: set(s.dropna().astype(str).unique())
                    ),
                    num_nans=raw.isnull().sum(axis=0),
                    max=raw.apply(
                        _descriptive_stat_reducer(max), result_type="expand"
                    ),
                    min=raw.apply(
                        _descriptive_stat_reducer(min), result_type="expand"
                    ),
                    mean=raw.apply(
                        _descriptive_stat_reducer(np.mean),
                        result_type="expand",
                    ),
                    stddev=raw.apply(
                        _descriptive_stat_reducer(np.std), result_type="expand"
                    ),
                )
            )
            .pipe(
                lambda df: df.assign(
                    num_distincts=df["samples_set"].apply(len)
                )
            )
        )

        # Get five random samples
        samples = (
            df[["samples_set"]]
            .apply(
                lambda row: RawDataSetParser._generate_samples(row.values[0]),
                axis=1,
                result_type="expand",
            )
            .rename(lambda i: f"sample{i + 1}", axis=1)
        )

        df = pd.concat([df, samples], sort=False, axis=1).reset_index(
            drop=True
        )

        return df

    @staticmethod
    def _generate_samples(s: set, n: int = 5) -> np.ndarray:
        """
        Generate `n` samples from set `s` without replacement.

        If there are less than `n` samples, than additional values will be sampled
        with replacement until the set reaches length `n`.
        """
        # Remove NaNs from set. This assumes that, other than NaNs, the set only contains
        # numeric and text members.
        without_nans = {x for x in s if x == x}

        # Attempt to sample without replacement
        if len(without_nans) >= n:
            return np.random.choice(list(without_nans), size=n, replace=False)
        elif len(without_nans) == 0:
            return np.array([np.nan] * n)
        else:
            # Add all members from set equally before randomly sampling remaining members
            samples = list(without_nans) * (n // len(without_nans))
            samples.extend(
                np.random.choice(
                    list(without_nans), size=(n - len(samples)), replace=False
                ).tolist()
            )

            # Randomize sample orders
            np.random.shuffle(samples)
            return samples

    def normalize_features(
        self, fitted_scaler: Type[RobustScaler]
    ) -> pd.DataFrame:
        """
        Normalize metafeatures for training and inference.

        Removes non-normalizable metafeatures before performing
        normalization using a fitted scaler that was used to train another model.

        Arguments:
            fitted_scaler {Type[RobustScaler]}
            -- A scaler that was fitted to the training data.

        Returns:
            pd.DataFrame -- Features for prediction.
        """
        X_to_normalize, X_to_retain = super()._select_metafeatures(
            self.test_metafeatures
        )
        X_to_normalize = pd.DataFrame(
            fitted_scaler.transform(X_to_normalize),
            columns=X_to_normalize.columns,
            index=X_to_normalize.index,
        )

        X = pd.concat([X_to_normalize, X_to_retain], axis=1)

        return X
