"""Common module utilities."""

import sys
import warnings

import numpy as np
import pandas as pd


PipelineStep = {"NAME": 0, "CLASS": 1, "COLS": 2}


def check_df(input_data, ignore_none=False, single_column=False):
    """Convert non dataframe inputs into dataframes (or series).

    Args:
        input_data (:obj:`pandas.DataFrame`, :obj:`numpy.ndarray`, list): input
            to convert
        ignore_none (bool): allow None to pass through check_df
        single_column (bool): check if frame is of a single column and return
            series

    Returns:
        :obj:`DataFrame <pandas.DataFrame>`: Converted and validated input \
            dataframes

    Raises:
        ValueError: Invalid input type
        ValueError: Input dataframe must only have one column

    """
    if input_data is None and ignore_none:
        return None

    ret_df = None
    if isinstance(input_data, pd.DataFrame):
        if len(input_data.columns) > len(set(input_data.columns)):
            warnings.warn(
                "Columns are not all uniquely named, automatically resolving"
            )
            input_data.columns = pd.io.parsers.ParserBase(
                {"names": input_data.columns}
            )._maybe_dedup_names(input_data.columns)
        ret_df = input_data
    elif isinstance(input_data, pd.Series):
        ret_df = input_data.to_frame()
    elif isinstance(input_data, np.ndarray) or isinstance(
        input_data, (list, tuple)
    ):
        ret_df = pd.DataFrame(input_data)
    else:
        raise ValueError(
            "Invalid input type, neither pd.DataFrame, pd.Series, np.ndarray, "
            "nor list"
        )

    if single_column and len(ret_df.columns) != 1:
        raise ValueError("Input Dataframe must have only one column")

    return ret_df


def check_module_installed(name):
    """Check whether a module is available for import.

    Args:
        name (str): module name

    Returns:
        bool: Whether the module can be imported

    """
    try:
        __import__(name)
    except ImportError:
        return False
    else:
        return True


def check_transformer_imports(printout=True):
    """Determine which transformers were automatically imported.

    Args:
        printout (bool, optional): Whether to output to stdout

    Returns:
        tuple(list): A tuple of the internal transformers and the external \
            transformers

    """
    from foreshadow.transformers import externals as exter
    from foreshadow.transformers import internals as inter

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


def debug():  # pragma: no cover # noqa: D202
    """Add pdb debugger on import.

    Utility to add pdb debugging to an entire file so that on error, the pdb
    utility is opened.
    """

    def _info(type, value, tb):
        # Source: https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error # noqa
        if hasattr(sys, "ps1") or not sys.stderr.isatty():
            sys.__excepthook__(type, value, tb)
        else:
            import traceback
            import pdb

            traceback.print_exception(type, value, tb)
            pdb.post_mortem(tb)

    sys.excepthook = _info
