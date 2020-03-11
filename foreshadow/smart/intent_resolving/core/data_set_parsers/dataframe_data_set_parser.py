"""Class defintion for the DataFrameDataSetParser concrete class."""

from typing import Callable, Generator, Tuple

import pandas as pd

from ..heuristics import has_zero_in_leading_decimals
from .lazy_dataframe_loader import LazyDataFrameLoader
from .raw_data_set_parser import RawDataSetParser


class DataFrameDataSetParser(RawDataSetParser):
    """
    Concrete class to extract metafeatures from a DataFrame.

    Attributes:
        raw {pd.DataFrame}
            -- Raw dataframe to analyze
        test_metafeatures {pd.DataFrame}
            -- Metafeatures extracted from the dataframe
        _DATASET_ID_PLACEHOLDER
            -- Placeholder string to conform with type requirements of
               `_create_raw_generator`.

        Refer to RawDataSetParser superclass for additional attributes.

    Methods:
        load_data_set -- Extract base features from the raw dataframe
        featurize_base -- Perform base featurization
        featurize_secondary -- Perform secondary featurization
        normalize_features -- Normalize the test metafeatures
    """

    def __init__(self, raw: pd.DataFrame):
        """
        Init function.

        Arguments:
            raw {pd.DataFrame} -- Raw dataframe to analyze
        """
        super().__init__()
        self.raw = DataFrameDataSetParser.__clean(raw)
        self._DATASET_ID_PLACEHOLDER = "dataset"

    @staticmethod
    def __clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts float columns whose decimals are zero, into int columns.

        Arguments:
            df {pd.DataFrame} -- Raw dataframe to be cleaned.

        Returns:
            pd.DataFrame -- A cleaned dataframe.
        """
        conditions = has_zero_in_leading_decimals(df)
        for i, col in enumerate(df.columns):
            if conditions[i]:
                df[col] = df[col].astype(int)
        return df

    def load_data_set(self) -> None:
        """
        Load data set.

        Dummy function since raw data set is already provided.
        """
        return

    def featurize_base(self) -> None:
        """Extract base features from DataFrame."""
        self.test_metafeatures = super()._extract_base_features(self.raw)

    def _create_raw_generator(
        self
    ) -> Generator[Tuple[str, Callable[[], pd.DataFrame]], None, None]:
        yield (self._DATASET_ID_PLACEHOLDER, LazyDataFrameLoader(df=self.raw))
