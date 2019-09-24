"""Foreshadow console wrapper."""
# flake8: noqa
# isort: noqa
import argparse
import json
import sys
import warnings

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from foreshadow.config import config
from foreshadow.estimators import AutoEstimator
from foreshadow.foreshadow import Foreshadow


def generate_model(args):  # noqa: C901
    """Process command line args and generate a Foreshadow model to fit.

    Args:
        args (list): A list of string arguments to process

    Returns:
        tuple: A tuple of `fs, X_train, y_train, X_test, y_test` which \
            represents the foreshadow model along with the split data.

    Raises:
        ValueError: if invalid file or invalid y.

    """
    parser = argparse.ArgumentParser(
        description="Peer into the future of a data science project"
    )
    parser.add_argument(
        "data", type=str, help="File path of a valid CSV file to load"
    )
    parser.add_argument(
        "target", type=str, help="Name of target column to predict in dataset"
    )
    parser.add_argument(
        "problem_type",
        default="classification",
        type=str,
        choices=["classification", "regression"],
        help="Problem type, choosing from classification or regression, "
        "default to classification.",
    )
    parser.add_argument(
        "--multiprocess",
        default=False,
        type=bool,
        help="Whether to enable multiprocessing on the dataset, useful for "
        "large datasets and/or computational heavy transformations.",
    )
    parser.add_argument(
        "--level",
        default=1,
        type=int,
        help="Level of fitting 1: All defaults 2: Feature engineering"
        "parameter search 3: Model parameter search"
        "using AutoSklearn or TPOT ",
    )
    parser.add_argument(
        "--method",
        default=None,
        type=str,
        help="Name of Estimator class from sklearn.linear_model to use."
        "Defaults to LogisticRegression for classification"
        "and LinearRegression for regression",
    )
    parser.add_argument(
        "--time",
        default=10,
        type=int,
        help="Time limit in minutes to apply to model"
        "parameter search. (Default 10)",
    )
    parser.add_argument(
        "--x_config",
        default=None,
        type=str,
        help="Path to JSON configuration file for X Preprocessor",
    )
    parser.add_argument(
        "--y_config",
        default=None,
        type=str,
        help="Path to JSON configuration file for y Preprocessor",
    )
    cargs = parser.parse_args(args)

    if cargs.level == 3 and cargs.method is not None:
        warnings.warn(
            "WARNING: Level 3 model search enabled. Method will be ignored."
        )

    if cargs.level != 3 and cargs.time != 10:
        warnings.warn(
            "WARNING: Time parameter not applicable "
            "to feature engineering. Must be in level 3."
        )

    try:
        df = pd.read_csv(cargs.data)
    except Exception:
        raise ValueError(
            "Failed to load file. Please verify it exists and is a valid CSV."
        )

    try:
        X_df = df.drop(columns=cargs.target)
        y_df = df[[cargs.target]]
    except Exception:
        raise ValueError("Invalid target variable")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2
    )

    if cargs.level == 1:
        # Default everything with basic estimator
        fs = Foreshadow(
            estimator=get_method(cargs.method, cargs.problem_type, y_train)
        )

    # elif cargs.level == 2:
    #     # Parameter search on all matched intents
    #
    #     if cargs.x_config is not None:
    #         try:
    #             with open(cargs.x_config, "r") as f:
    #                 X_search = Preprocessor(from_json=json.load(f))
    #         except Exception:
    #             raise ValueError(
    #                 "Could not read X config file {}".format(cargs.x_config)
    #             )
    #         print("Reading config for X Preprocessor")
    #     else:
    #         X_search = search_intents(X_train)
    #         print("Searching over valid intent space for X data")
    #
    #     if cargs.y_config is not None:
    #         try:
    #             with open(cargs.y_config, "r") as f:
    #                 y_search = Preprocessor(from_json=json.load(f))
    #         except Exception:
    #             raise ValueError(
    #                 "Could not read y config file {}".format(cargs.y_config)
    #             )
    #         print("Reading config for y Preprocessor")
    #     else:
    #         y_search = search_intents(y_train, y_var=True)
    #         print("Searching over valid intent space for y data")
    #
    #     # If level 3 also do model parameter search with AutoEstimator
    #     # Input time limit into Foreshadow to be passed into AutoEstimator
    #
    #     fs = Foreshadow(
    #         X_preparer=X_search,
    #         y_preparer=y_search,
    #         estimator=get_method(cargs.method, y_train),
    #         optimizer=GridSearchCV,
    #     )
    #
    elif cargs.level == 3:
        # Default intent and advanced model search using 3rd party AutoML

        estimator = AutoEstimator(problem_type=cargs.problem_type, auto="tpot")
        estimator.configure_estimator(y_train)

        # TODO move this into the configure_estimator method
        # TODO "max_time_mins" is an argument for the TPOT library. We cannot
        # TODO assign it based on the problem type here. For testing purpose,
        # TODO I'm going to hardcode it for TPOT.
        # kwargs = (
        #     "max_time_mins"
        #     if estimator.problem_type == "regression"
        #     else "time_left_for_this_task"
        # )
        kwargs = "max_time_mins"
        estimator.estimator_kwargs = {
            kwargs: cargs.time,
            **estimator.estimator_kwargs,
        }

        fs = Foreshadow(estimator=estimator)

    else:
        raise ValueError("Invalid Level. Only levels 1 and 3 supported.")

    if cargs.multiprocess:
        config.set_multiprocess(True)
        print("multiprocessing enabled.")

    return fs, X_train, y_train, X_test, y_test


def execute_model(fs, X_train, y_train, X_test, y_test):
    """Execute the model produced by `generate_model()`.

    Also, exports the data to json and returns the exported json object
    containing the results and the serialized Foreshadow object. Also, prints
    simple model accuracy metrics.

    Args:
        fs (foreshadow.Foreshadow): An unfit foreshadow object.
        X_train (:obj:`DataFrame <pandas.DataFrame>`): The X train data.
        X_test (:obj:`DataFrame <pandas.DataFrame>`): The X test data.
        y_train (:obj:`DataFrame <pandas.DataFrame>`): The y train data.
        y_test (:obj:`DataFrame <pandas.DataFrame>`): The y test data.

    Returns:
        dict: A dictionary with the following keys `X_Model`, `X_Summary`, \
            `y_model`, and `y_summary` which each represent the serialized \
            and summarized forms of each of those steps.

    """
    print("Fitting final model...")
    fs.fit(X_train, y_train)

    print("Scoring final model...")
    score = fs.score(X_test, y_test)

    print("Final Results: ")
    print(score)

    fs.to_json("foreshadow.json")
    print(
        "Serialized foreshadow pipeline has been saved to foreshadow.json. "
        "Refer to docs to read and process."
    )
    # TODO serialize the foreshadow object and summarize the X and y stats.
    # Store final results
    # all_results = {
    #     "X_Model": fs.X_preparer.serialize(),
    #     # "X_Summary": fs.X_preparer.summarize(X_train),
    #     "y_Model": fs.y_preparer.serialize(),
    #     # "y_summary": fs.y_preparer.summarize(y_train),
    # }
    # return all_results


def cmd():  # pragma: no cover
    """Entry point to foreshadow via console command.

    Uncovered as this function only serves to be executed manually.
    """
    model = generate_model(sys.argv[1:])
    execute_model(*model)


def get_method(method, problem_type, y_train):
    """Determine what estimator to use.

    Uses set of X data and a passed argument referencing an
    `BaseException <sklearn.base.BaseEstimator>` class.

    Args:
        method (str): model name
        problem_type (str): problem type, classification or regression
        y_train (:obj:`DataFrame <pandas.DataFrame>`): The response variable
            data.

    Returns:
        Estimator

    Raises:
        ValueError: if invalid method is chosen

    """
    if method is not None:
        try:
            mod = __import__(
                "sklearn.linear_model", globals(), locals(), ["object"], 0
            )
            cls = getattr(mod, method)
            return cls()
        except Exception:
            raise ValueError(
                "Invalid method. {} is not a valid "
                "estimator from sklearn.linear_model".format(method)
            )
    else:
        return (
            LinearRegression()
            if problem_type == "regression"
            else LogisticRegression()
        )


# def search_intents(X_train, y_var=False):
#     """Hyper-parameter searches across intents.
#
#     Args:
#         X_train (:obj:`DataFrame <pandas.DataFrame>`): The X train data.
#         y_var (bool, optional): specifies whether the
#             :obj:`Preprocessor <foreshadow.Preprocessor>` is of a y variable.
#
#     Returns:
#         Preprocessor
#
#     """
#     proc = Preprocessor(y_var=y_var)
#
#     proc.fit(X_train)
#
#     result = proc.serialize()
#
#     space = {
#         "columns": {
#             k: {"intent": v["intent"]} for k, v in result["columns"].items()
#         },
#         "combinations": [
#             {
#                 "columns.{}.intent".format(k): str(
#                     [
#                         i
#                         for i in v["all_matched_intents"]
#                         if i != "GenericIntent"
#                     ]
#                 )
#                 for k, v in result["columns"].items()
#             }
#         ],
#     }
#
#     return Preprocessor(from_json=space, y_var=y_var)
