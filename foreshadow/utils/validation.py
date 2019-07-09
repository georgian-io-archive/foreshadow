"""Common module utilities."""

import warnings
from importlib import import_module

import numpy as np
import pandas as pd


PipelineStep = {"NAME": 0, "CLASS": 1, "COLS": 2}

INTERNALS_SOURCE = "foreshadow.transformers.internals"
EXTERNALS_SOURCE = "foreshadow.transformers.externals"


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
        internals = import_module(INTERNALS_SOURCE)
        externals = import_module(EXTERNALS_SOURCE)

        if hasattr(internals, class_name):
            module = internals
        elif hasattr(externals, class_name):
            module = externals
        else:
            raise ValueError(
                (
                    "Could not find transformer {} in neither "
                    "foreshadow.transformers.internals nor "
                    "foreshadow.transformers.externals"
                ).format(class_name)
            )

    return getattr(module, class_name)
