"""Common module utilities."""

import warnings
from collections import OrderedDict
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin


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
            "Invalid input type: {} is not pd.DataFrame, "
            "pd.Series, "
            "np.ndarray, "
            "nor list".format(type(input_data))
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


def get_transformer(class_name, source_lib=None):
    """Get the transformer class from its name.

    Note:
        In case of name conflict, internal transformer is preferred over
        external transformer import.

    Args:
        class_name (str): The transformer class name
        source_lib (str): The string import path if known

    Returns:
        Imported class

    Raises:
        ValueError: If class_name could not be found in internal or external
            transformer library pathways.

    """
    if source_lib is not None:
        module = import_module(source_lib)
    else:
        sources = OrderedDict(
            (source, import_module(source))
            for source in [
                "foreshadow.transformers.internals",
                "foreshadow.transformers.externals",
                "foreshadow.transformers.smart",
            ]
        )

        for v in sources.values():
            if hasattr(v, class_name):
                module = v
                break
        else:
            raise ValueError(
                "Could not find transformer {} in {}".format(
                    class_name, ", ".join(sources.keys())
                )
            )

    return getattr(module, class_name)


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
        bool: Whether or not the transformer is wrapped.

    """
    return "make_pandas_transformer" in repr(transformer.__class__)
