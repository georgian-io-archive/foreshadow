"""Class definition for the BaseFeaturizer ABC."""

from abc import ABC, abstractmethod
from typing import List, Sequence

import pandas as pd


class BaseFeaturizer(ABC):
    """
    Abstract base class to perform secondary featurization.

    Attributes:
        method {str}
            -- Description of featurizer.
        normalizable {bool}
            -- Whether the generated feature should be normalized. (default: {False})
        sec_feature_names {List[str]}
            -- Names for secondary metafeatures
        sec_metafeatures {pd.DataFrame}
            -- Stores extracted secondary metafeatures.
        sec_test_metafeatures {pd.DataFrame}
            -- Stores extracted secondary metafeatures from the test dataframe.

    Abstract methods:
        featurize -- Perform secondary featurization.
    """

    def __init__(self, method: str, normalizable: bool):
        """
        Init function.

        Arguments:
            method {str} -- Description of featurizer.
            normalizable {bool} -- Whether the generated feature should be
                                   normalized. (default: {False})
        """
        self.method = method
        self.normalizable = normalizable

        self.sec_feature_names: Sequence[str] = None
        self.sec_metafeatures: pd.DataFrame = None
        self.sec_test_metafeatures: pd.DataFrame = None

    @abstractmethod
    def featurize(self):
        """Perform secondary featurization."""
        raise NotImplementedError

    @staticmethod
    def _mark_nonnormalizable(
        names: Sequence[str], mark: str = "*", normalizable: bool = True
    ) -> List[str]:
        """
        Mark features to be not normalized but to be used for training / testing.

        Append `mark` onto secondary metafeaeture names to indicate to
        DataSetParser that these objects are not to be normalized.

        Arguments:
            names {Sequence[str]} -- List of metafeature column names.

        Keyword Arguments:
            mark {str} -- String to concatenate to each column name. (default: {'*'})
            normalizable {bool} -- If False, return original `names`. (default: {True})
        """
        if normalizable:
            return names
        return [n + mark for n in names]

    def __delattr__(self, attr: str):
        """
        Set provided attribute to None.

        Arguments:
            attr {str} -- Attribute to reset.
        """
        self.__dict__[attr] = None
