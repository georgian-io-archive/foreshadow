"""
Utils used throughout foreshadow
"""

import warnings

import numpy as np
import pandas as pd


def check_df(input_data):
    """Convert non dataframe inputs into dataframes.
    
    Args:
        input_data (list or pandas.DataFrame or numpy.ndarray): input to convert

    Returns:
        (pandas.Dataframe): Convereted and validated input dataframes
    """
    if isinstance(input_data, pd.DataFrame):
        if len(input_data.columns) > len(set(input_data.columns)):
            warnings.warn("Columns are not all uniquely named, automatically resolving")
            input_data.columns = pd.io.parsers.ParserBase(
                {"names": input_data.columns}
            )._maybe_dedup_names(input_data.columns)
        return input_data
    elif isinstance(input_data, np.ndarray) or isinstance(input_data, list):
        return pd.DataFrame(input_data)
    else:
        raise ValueError(
            "Invalid input type, neither pd.DataFrame, np.ndarray, nor list"
        )
