"""Class definition for the DataSetParser ABC and FeaturizerMixin."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Generator, List, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class FeaturizerMixin:
    """Mixin to provide secondary featurization functionality."""

    def featurize_secondary(self):
        """
        Perform secondary featurization.

        Sequentially trigger each featurizer to extract secondary features.
        The extracted secondary metafeatures are stored in each featurizer's
        `sec_metafeatures` and `sec_test_metafeatures` attributes.

        These extracted metafeatures will then be collected and appended column-wise
        to the `metafeature` and `test_metafeature` attributes of the DataSetParser
        subclass instance.
        """
        for featurizer in self.featurizers:
            if type(featurizer).__name__ == "RawDataSetFeaturizerViaLambda":
                featurizer.featurize(
                    self._create_raw_generator(),
                    keys=self.metafeatures,
                    test_keys=self.test_metafeatures,
                    multiprocess=self._multiprocess_raw_secondary,
                )
            else:
                featurizer.featurize(
                    meta_df=self.metafeatures,
                    test_meta_df=self.test_metafeatures,
                )

        self.__add_secondary_metafeatures()

    def __add_secondary_metafeatures(self):
        """Add secondary features to the training and test metafeature attributes."""
        # Get secondary feature names
        if self.metafeatures is not None:
            sec_feature_names = list(self.metafeatures) + [
                name
                for featurizer in self.featurizers
                for name in featurizer.sec_feature_names
            ]
        elif self.test_metafeatures is not None:
            sec_feature_names = list(self.test_metafeatures) + [
                name
                for featurizer in self.featurizers
                for name in featurizer.sec_feature_names
            ]

        if self.metafeatures is not None:
            sec_metafeatures = [x.sec_metafeatures for x in self.featurizers]
            self.metafeatures = pd.concat(
                [self.metafeatures, *sec_metafeatures],
                axis=1,
                ignore_index=True,
            )
            self.metafeatures.columns = sec_feature_names

        if self.test_metafeatures is not None:
            sec_test_metafeatures = [
                x.sec_test_metafeatures for x in self.featurizers
            ]
            self.test_metafeatures = pd.concat(
                [self.test_metafeatures, *sec_test_metafeatures],
                axis=1,
                ignore_index=True,
            )
            self.test_metafeatures.columns = sec_feature_names


class DataSetParser(ABC, FeaturizerMixin):
    """
    Abstract base class to load and extract metafeatures from raw data sets.

    FeaturizerMixin provides the `.featurize` method.


    Instance attributes:
        src {Path}
            -- Path to data set file on disk.
        metafeatures {pd.DataFrame}
            -- Metafeatures extracted from the raw data set. Each metafeature
               row corresponds to a feature column in the raw data set.
        labels {pd.Series}
            -- Label corresponding to each metafeature.
        test_src {Path}
            -- Optional path to test raw data set file on disk. This attribute
               applies more to the subclasses of MetaDataSetParser.
        test_metafeatures {pd.DataFrame}
            -- Optional metafeatures extracted from the test raw data set.
        test_labels {pd.Series}
            -- Optional labels corresponding to each test metafeature row.
        scaler {RobustScaler}
            -- A scaler to handle normalize metafeatures before serving them
               for training.
        featurizers: {List}
            -- A list of featurizers that performs secondary metafeaturizations.

    Class attributes:
        NUM_BASE_METAFEATURES {int}
            -- Number of base metafeatures.
               Used to separate base and secondary metafeatures.

    Abstract methods:
        load_data_set
            -- Load the data set and perform necessarily cleaning and parsing.
        featurize_base
            -- Featurize base metafeatures.
        normalize_features
            -- Performs normalization on the metafeatures and test metafeatures
               (if provided).
        _create_raw_generator
            -- Returns a generator of raw data sets. This supports the
               MetaDataSetFeaturizerViaLambda class functionality.
    """

    NUM_BASE_METAFEATURES = (
        7
    )  # Includes (total_val, min, max, mean, std, num_nans, num_distincts)

    def __init__(self):
        """Init function."""
        self.src: Path = None
        self.labels: pd.Series = None
        self.metafeatures: pd.DataFrame = None

        self.test_src: Path = None
        self.test_labels: pd.Series = None
        self.test_metafeatures: pd.DataFrame = None

        self.scaler: Type[RobustScaler] = None

        self.featurizers: List = []
        self._multiprocess_raw_secondary: bool = False  # Multiprocessing of raw dataframe(s)

    @abstractmethod
    def load_data_set(self):
        """Load data set from source."""
        raise NotImplementedError

    @abstractmethod
    def featurize_base(self):
        """Featurize base metafeatures."""
        raise NotImplementedError

    @abstractmethod
    def normalize_features(self):
        """Normalize metafeatures for training."""
        raise NotImplementedError

    @abstractmethod
    def _create_raw_generator(
        self
    ) -> Generator[Tuple[str, Callable[[], pd.DataFrame]], None, None]:
        raise NotImplementedError

    def _select_metafeatures(
        self, df: pd.DataFrame, mark: str = "*"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select metafeatures to normalize and to retain for training.

        The following criteria is used.

        Metafeatures to:
            - normalize: Numerical columns
            - not normalize but retain for training: Features whose title ends with `mark`.

        Remainder metafeatures are dropped.

        Note:
        Columns are tracked by indices instead of names to avoid problems when
        there are duplicated columnn names.

        Arguments:
            df {pd.DataFrame}
                -- Metafeatures dataframe.
            mark {str}
                -- Character to append to names of columns that should not be
                   normlized but retained for training.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]
                -- (metafeatures_to_normalize, metafeatures_to_retain)
        """
        idx_to_normalize: List[int] = []
        idx_to_retain: List[int] = []

        IGNORE_COLS = (
            "attribute_name",  # Already represented as ngrams
            "sample",  # Ignore sample columns which may be of type int
            "total_val",  # Intent prediction should not be based on # data points
            "num_distincts",  # Use `normalized_distinct_rate` instead
            "num_nans",  # Captured in `nan_rate`
        )

        for i, col in enumerate(df.columns):
            if col in IGNORE_COLS:
                continue

            # Save columns that are either numeric or that have been marked
            # into appropriate groups
            if col[-1] == "*":
                idx_to_retain.append(i)
            elif self._is_numeric(df.iloc[:, i]):
                idx_to_normalize.append(i)

        features_to_normalize = df.iloc[:, idx_to_normalize]
        features_to_retain = df.iloc[:, idx_to_retain]

        return features_to_normalize, features_to_retain

    def _is_numeric(self, series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series)

    @staticmethod
    def _split_features_and_labels(
        mds: pd.DataFrame, label_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split features and labels.

        Arguments:
            mds {pd.DataFrame} -- MetaDataSet.
            label_col {str} -- Column containing labels in the MetaDataSet.

        Returns:
            Tuple[pd.DataFrame, pd.Series] -- (features, labels) tuple.
        """
        return mds.drop(label_col, axis=1), mds[label_col]
