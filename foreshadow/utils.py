"""
Utils used throughout foreshadow
"""

import warnings

import numpy as np
import pandas as pd

PipelineStep = {"NAME": 0, "CLASS": 1, "COLS": 2}


def check_df(input_data, ignore_none=False, single_column=False):
    """Convert non dataframe inputs into dataframes. (or series)
    
    Args:
        input_data (:obj:`pandas.DataFrame`, :obj:`numpy.ndarray`, list):
            input to convert
        ignore_none (bool): allow None to pass through check_df
        single_column (bool): check if frame is of a single column and return series

    Returns:
        :obj:`pandas.DataFrame`: Converted and validated input dataframes
    """

    if input_data is None and ignore_none:
        return None

    ret_df = None
    if isinstance(input_data, pd.DataFrame):
        if len(input_data.columns) > len(set(input_data.columns)):
            warnings.warn("Columns are not all uniquely named, automatically resolving")
            input_data.columns = pd.io.parsers.ParserBase(
                {"names": input_data.columns}
            )._maybe_dedup_names(input_data.columns)
        ret_df = input_data
    elif isinstance(input_data, pd.Series):
        ret_df = input_data.to_frame()
    elif isinstance(input_data, np.ndarray) or isinstance(input_data, (list, tuple)):
        ret_df = pd.DataFrame(input_data)
    else:
        raise ValueError(
            "Invalid input type, neither pd.DataFrame, pd.Series, np.ndarray, nor list"
        )

    if single_column and len(ret_df.columns) != 1:
        raise ValueError("Input Dataframe must have only one column")

    return ret_df


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
