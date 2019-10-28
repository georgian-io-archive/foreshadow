"""Class defintion for the DataFrameDataSetParser concrete class."""

from typing import Generator, Tuple

import pandas as pd

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
        self.raw = raw
        self._DATASET_ID_PLACEHOLDER = "dataset"

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
    ) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        yield (self._DATASET_ID_PLACEHOLDER, self.raw)
