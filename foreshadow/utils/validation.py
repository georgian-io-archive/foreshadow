"""Common module utilities."""

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import VectorizerMixin

from foreshadow.base import BaseEstimator, TransformerMixin


PipelineStep = {"NAME": 0, "CLASS": 1, "COLS": 2}


def check_series(input_data):
    """Convert non series inputs into series.

    This is function is to be used in situations where a series is expected but
    cannot be guaranteed to exist. For example, this function is used in
    the metrics package to perform computations on a column using functions
    that only work with series.

    Note:
        This is not to be used in transformers as it will break the standard
        that enforces only DataFrames as input and output for those objects.

    Args:
        input_data (iterable): The input data

    Returns:
        pandas.Series

    Raises:
        ValueError: If the data could not be processed
        ValueError: If the input is a DataFrame and has more than one column

    """
    ret_ser = None

    if isinstance(input_data, pd.DataFrame):
        ret_ser = input_data.T.squeeze()
        if not isinstance(ret_ser, pd.Series):
            raise ValueError("Input DataFrame has more than one column")
    elif isinstance(input_data, pd.Series):
        pass
    elif isinstance(input_data, np.ndarray) or isinstance(
        input_data, (list, tuple)
    ):
        ret_ser = pd.Series(input_data)
    else:
        raise ValueError(
            "Invalid input type: {} is not pd.DataFrame, "
            "pd.Series, "
            "np.ndarray, "
            "nor list".format(type(input_data))
        )

    return ret_ser


def check_df(
    input_data, ignore_none=False, single_column=False, single_or_empty=False
):
    """Convert non dataframe inputs into dataframes.

    Args:
        input_data (:obj:`pandas.DataFrame`, :obj:`numpy.ndarray`, list): input
            to convert
        ignore_none (bool): allow None to pass through check_df
        single_column (bool): check if frame is of a single column and return
            series
        single_or_empty (bool): check if the frame is a single column or an
            empty DF.

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
            "Invalid input type: {} is not pd.DataFrame, "
            "pd.Series, "
            "np.ndarray, "
            "nor list".format(type(input_data))
        )

    if single_column and len(ret_df.columns) != 1:
        raise ValueError("Input Dataframe must have only one column")
        # TODO custom error.
    if single_or_empty and (len(ret_df.columns) != 1 and not ret_df.empty):
        raise ValueError(
            "Input Dataframe must have only one column or be " "empty."
        )
    return ret_df


def check_module_installed(name):
    """Check whether a module is available for import.

    Args:
        name (str): module name

    Returns:
        bool: Whether the module can be imported

    """
    from importlib import import_module

    try:
        import_module(name)
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
    import foreshadow.concrete as conc

    if printout:
        print(
            "Loaded {} transformer plugins:\n{}".format(
                len(conc.__all__), conc.__all__
            )
        )

    return conc.__all__


def is_transformer(value, method="isinstance"):
    """Check if the class is a transformer class.

    Args:
        value: Class or instance
        method (str): Method of checking. Options are `'issubclass'` or
            `'isinstance'`

    Returns:
        True if transformer, False if not.

    Raises:
        ValueError: if method is neither issubclass or isinstance

    """
    choices = {"issubclass": issubclass, "isinstance": isinstance}
    try:
        validation = choices.get(method)
    except KeyError:
        raise ValueError(
            "Could not find {}, options are {}".format(method, choices.keys())
        )

    if validation(value, BaseEstimator) and (
        validation(value, TransformerMixin)
        or validation(value, VectorizerMixin)
    ):
        return True
    return False


def is_wrapped(transformer):
    """Check if a transformer is wrapped.

    Args:
        transformer: A transformer instance

    Returns:
        bool: True if transformer is wrapped, otherwise False.

    """
    return hasattr(transformer, "is_wrapped")
