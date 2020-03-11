"""Class definition for LazyDatarameLoader."""
import re
from hashlib import md5
from pathlib import Path
from string import ascii_letters, digits
from typing import List, Optional, Pattern, Union

import pandas as pd


class LazyDataFrameLoader:
    """
    Loads a dataframe lazily when called.

    This avoids the bottleneck of loading dataframes during model training
    and feature engineering, by parallelizing the loading when
    RawDataSetFeaturizerViaLambda.__slow_featurize is called with
    multiprocessing.


    Attributes:
        df {Optional[pd.DataFrame]}
            -- A raw dataframe. (default: {None})
        csv_path {Optional[Union[str, Path]]}
            -- Path to a dataframe (default: {None})

    Methods:
        __call__ -- Loads a dataframe
    """

    def __init__(
        self,
        *,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[Union[str, Path]] = None,
        remove_id_substring: Optional[bool] = False,
    ):
        """Init function.

        Only `df` or `csv_path` can be specified.

        Keyword Arguments:
            df {Optional[pd.DataFrame]}
                -- A raw dataframe. (default: {None})
            csv_path {Optional[Union[str, Path]]}
                -- Path to a dataframe (default: {None})
            remove_id_substring {Optional[bool]}
                -- If True, replaces 'id'-like substrings in column names
                   of dataframes (default: False)
            pattern {Pattern}
                -- Compiled regex pattern

        Raises:
            ValueError: When both `df` and `csv_path` are specified
            TypeError: When `df` is not a pd.DataFrame
            TypeError: When `csv_path` is not a str or a Path
            ValueError: When `csv_path` is not a csv file
        """
        # Ensure only one of two optional keyword arguments is provided
        if (df is not None) == (csv_path is not None):
            raise ValueError("Only one of `df` or `csv_path` can be provided.")
        if (df is not None) and not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Expecting `df` of type pd.DataFrame. " f"Got type {type(df)}."
            )
        if csv_path is not None:
            if not (isinstance(csv_path, str) or isinstance(csv_path, Path)):
                raise TypeError(
                    "Expecting `csv_path` of type str or Path. "
                    f"Got type {type(csv_path)}."
                )
            if str(csv_path)[-3:].lower() != "csv":
                raise ValueError("A CSV file is expected for `csv_path`.")

        # Either one of this will have a non-None value
        self.df = df
        self.csv_path = csv_path
        self.remove_id_substring = remove_id_substring
        self.pattern = re.compile("_?[iI][dD]")

    def __call__(self) -> pd.DataFrame:
        """Loads the provided (path to a) dataframe."""
        if self.df is not None:
            result = self.df
        else:
            result = pd.read_csv(
                self.csv_path, encoding="latin", low_memory=False
            )

        # Optionally clean 'id'-like substrings from column names
        if self.remove_id_substring:
            result.columns = self._replace_id_in_col_name(result)
        return result

    def _replace_id_in_col_name(self, columns: List[str]) -> List[str]:
        sub = HashSubstituter()
        replacement = [
            re.sub(self.pattern, sub.substitute(col), col) for col in columns
        ]

        # Hard fails if there is a collision post-ID-substitution
        if pd.Series(replacement).nunique() < len(replacement):
            raise ValueError(
                "Collision occurred when substituting ID-columns for"
                f"{self.df.head() if self.df is not None else self.csv_path}\n"
                f"Values post-substitution: {replacement}."
            )
        return replacement


class HashSubstituter:
    """
    Substitute a string with a character deterministically using md5 hashing.

    Attributes:
        chars -- List of characters to sample from as substitute
    """

    def __init__(self):
        self.chars = list(ascii_letters + digits)

    def substitute(self, s: str) -> str:
        """
        Get a random alphabetical or numerical character.

        This is a pure function — results are deterministic,
        and uses the md5 hash function.

        Returns:
            str -- A character from self.chars
        """
        hashed = int(md5(s.encode()).hexdigest(), 16)
        return self.chars[hashed % len(self.chars)]
