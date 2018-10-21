"""
Utils used throughout foreshadow
"""

import warnings

import numpy as np
import pandas as pd


def check_df(input_data, ignore_none=False):
    """Convert non dataframe inputs into dataframes.
    
    Args:
        input_data (:obj:`pandas.DataFrame`, :obj:`numpy.ndarray`, list):
            input to convert

    Returns:
        :obj:`pandas.DataFrame`: Converted and validated input dataframes
    """

    if input_data is None and ignore_none:
        return None

    if isinstance(input_data, pd.DataFrame):
        if len(input_data.columns) > len(set(input_data.columns)):
            warnings.warn("Columns are not all uniquely named, automatically resolving")
            input_data.columns = pd.io.parsers.ParserBase(
                {"names": input_data.columns}
            )._maybe_dedup_names(input_data.columns)
        return input_data
    elif isinstance(input_data, pd.Series):
        return input_data.to_frame()
    elif isinstance(input_data, np.ndarray) or isinstance(input_data, list):
        return pd.DataFrame(input_data)
    else:
        raise ValueError(
            "Invalid input type, neither pd.DataFrame, pd.Series, np.ndarray, nor list"
        )


def check_module_installed(name):
    """Checks whether a module is available for import"""
    try:
        __import__(name)
    except ImportError:
        return False
    else:
        return True


def check_transformer_imports(printout=True):
    """Determines which transformers were automatically imported"""

    from .transformers import externals as exter
    from .transformers import internals as inter

    if printout:
        print(
            "Loaded {} internals transformer plugins:\n{}".format(
                len(inter.classes), inter.classes
            )
        )
        print(
            "Loaded {} externals transformer plugins\n{}".format(
                len(exter.classes), exter.classes
            )
        )

    return inter.classes, exter.classes
