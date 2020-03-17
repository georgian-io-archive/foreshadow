"""
Definitions for classes that extract metafeatures from the raw data set(s).

Classes here take a user-defined function to apply to a raw dataframe to extract
the secondary metafeatures.

Class defintions included here are for:
    -- RawDataSetLambdaTransformer
    -- RawDataSetLambdaTransformerBuilder
    -- RawDataSetFeaturizerViaLambda and
    -- RawDataSetFeaturizerViaLambdaBuilder
"""

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Type

import pandas as pd
from tqdm import tqdm

from .. import io
from .base_featurizer_via_lambda import BaseFeaturizerViaLambda


class RawDataSetLambdaTransformer(BaseFeaturizerViaLambda):
    """
    Concrete class to transform a raw data set into a metafeature.

    Is a component to RawDataSetFeaturizerViaLambda.

    Attributes:
        Refer to superclass for attributes.

    Methods:
        featurize -- Perform secondary featurization
    """

    def __init__(
        self,
        method: str,
        callable_: Callable[[pd.DataFrame], pd.Series],
        normalizable: bool,
    ):
        """Init function inherited from superclass."""
        super().__init__(
            method=method, callable_=callable_, normalizable=normalizable
        )

    def featurize(self, raw: pd.DataFrame) -> pd.Series:
        """
        Extract numerical feature from raw dataframe.

        Arguments:
            raw {pd.DataFrame} -- Raw dataframe.

        Returns:
            pd.Series -- Extracted numerical features.
        """
        feature = self._callable(raw)
        feature.name = self.method
        return feature


def _featurize_raw_dataframe(
    data: Tuple[str, Callable[[], pd.DataFrame]],
    featurizers: List[Type[RawDataSetLambdaTransformer]],
) -> pd.DataFrame:
    """
    Applies RawDataSetLambdaTransformers.

    Extracted metafeatures are saved in a dataframe for each raw data set.

    To enable pickling during multiprocessing, this function has to be a module
    level function, instead of an instance method.

    Arguments:
        data {Tuple[str, Callable[[], pd.DataFrame]]}
            -- A dataset identifier string and a raw dataframe function from which
                metafeatures are extracted from.

    Returns:
        pd.DataFrame -- Extracted metafeatures from raw dataframe.
    """
    dataset = data[0]
    df = data[1]()

    metafeatures = (
        pd.concat([f.featurize(df) for f in featurizers], axis=1)
        .reset_index()
        .rename({"index": "attribute_name"}, axis=1)
    )
    metafeatures["dataset"] = dataset

    return metafeatures


class RawDataSetFeaturizerViaLambda:
    """
    Extract (multiple) metafeatures from a raw data set.

    Is to be used as a component to DataSetParser featurizers attributes,
    just like all the other featurizers subclasses.

    This class is composed of (multiple) RawDataSetLambdaTransformer.
    Each time a raw data set is loaded, this class will apply all its
    RawDataSetLambdaTransformers to the raw data set. This setup avoids loading
    the same raw data set repeatedly for each RawDataSetLambdaTransformer, since
    loading (large) raw data sets is the bottleneck in this operation.

    Attributes:
        featurizers {List[Type[RawDataSetLambdaTransformer]]}
            -- RawDataSetLambdaTransformers to apply to the raw data set(s)
        fast_load {bool}
            -- Load the pre-computed metafeatures from a previous run if they
               exist at `save_dir`.
        n_rows {Optional[Tuple[Optional[int],Optional[int]]]}
                -- Number of rows in the training and test metafeatures attributes
                   of DataSetParser subclass. Required if `fast_load` is True.
                   Refer to __init__ docs for concrete example (default: {None})
        save_dir {Optional[Path]}
                -- Directory to save or load computed secondary metafeatures.
                   Must be a pickle file. Required if `fast_load` is True.
                   (default: {None})

    Methods:
        featurize -- Extract secondary metafeatures from the raw data set(s).
    """

    def __init__(
        self,
        featurizers: List[Type[RawDataSetLambdaTransformer]],
        fast_load: bool = False,
        n_rows: Optional[Tuple[Optional[int], Optional[int]]] = None,
        save_dir: Optional[Path] = None,
    ):
        """
        Init function.

        Arguments:
            featurizers {List[Type[RawDataSetLambdaTransformer]]}
                -- RawDataSetLambdaTransformers to apply to the raw data set(s)
            fast_load {bool}
                -- Load the pre-computed metafeatures from a previous run if
                   they exist at `save_dir`.

        Keyword Arguments:
            n_rows {Optional[Tuple[Optional[int],Optional[int]]]}
                -- Number of rows in the training and test metafeatures attributes
                   of DataSetParser subclass. Required if `fast_load` is True.
                   (default: {None})

                   `n_rows` here is checked against the `n_rows` saved in the
                   pickle file at `save_dir`. `fast_load` will not trigger if
                   these values do not match.

                   If the (test) metafeature is None, the number of rows
                   defined to be None.

                   Example:
                        data_set_parser.metafeatures = None
                        data_set_parser.test_metafeatures = [pd.DataFrame with 30 rows]
                        # In this case, n_rows is (None, 30)

            save_dir {Optional[Path]}
                -- Directory to save or load computed secondary metafeatures.
                   Must be a pickle file. Required if `fast_load` is True.
                   (default: {None})

        Raises:
            ValueError -- If `save_dir` does not point to a pickle file.
            ValueError -- If `save_dir` and `n_rows` are not provided when
                          `fast_load` is True.
        """
        self.featurizers = featurizers

        self.sec_feature_names: List[str] = None
        self.sec_metafeatures: pd.DataFrame = None
        self.sec_test_metafeatures: pd.DataFrame = None

        self.save_dir = None
        self.n_rows = n_rows
        self.fast_load = fast_load

        if save_dir is not None:
            save_dir = Path(save_dir)

            if save_dir.suffix != ".pkl":
                raise ValueError(
                    f"Pickle file expected for `save_dir`. Got {save_dir}."
                )

            # If save_dir passes error checks, ensure directory exist and
            # set `save_dir` attribute
            save_dir.parent.mkdir(exist_ok=True, parents=True)
            self.save_dir = save_dir

        if self.fast_load and (self.save_dir is None or self.n_rows is None):
            raise ValueError(
                "Unable to perform fast load if "
                "`save_dir` and `n_rows` is not provided."
            )

    def featurize(
        self,
        raw_gen: Generator[Tuple[str, Callable[[], pd.DataFrame]], None, None],
        keys: Optional[pd.DataFrame] = None,
        test_keys: Optional[pd.DataFrame] = None,
        multiprocess: bool = False,
    ) -> None:
        """
        Extract secondary metafeatures from raw data set(s).

        At `key` and/or `test_keys` must be provided.

        Arguments:
            raw_gen {Generator[Tuple[str, Callable[[], pd.DataFrame]], None, None]}
                -- A raw dataframe generator.

        Keyword Arguments:
            keys {Optional[pd.DataFrame]}
                -- Metafeatures containing `dataset` and `attribute_name`.
                   (default: {None})
            test_keys {Optional[pd.DataFrame]}
                -- Test metafeatures contaiing `dataset` and `attribute_name`.
                   (default: {None})
            multiprocess {bool}
                -- If True, multiprocesses .__slow_featurize. This feature is
                   intended to shorten the model training cycle when new
                   features are included. (default: {False})

        Raises:
            ValueError -- Either `keys` or `test_keys` must be provided.
        """
        if keys is None and test_keys is None:
            raise ValueError(
                "At least one keyword argument "
                "(`keys`, `test_keys`) must be provided."
            )

        to_skip = [False] * len(self.featurizers)
        if (
            self.fast_load
            and self.save_dir.is_file()
            and self.__is_same_file()
        ):
            print("Found saved raw features. Quick loading instead...")
            self.__fast_featurize()
        else:
            self.__slow_featurize(
                raw_gen=raw_gen,
                keys=keys,
                test_keys=test_keys,
                multiprocess=multiprocess,
            )

        self.__get_feature_names()

    def __fast_featurize(self) -> None:
        """Quick load the same pre-computed secondary metafeatures from `save_dir`."""
        data = io.from_pickle(self.save_dir)
        self.sec_metafeatures = data["metafeatures"]
        self.sec_test_metafeatures = data["test_metafeatures"]

    def __slow_featurize(
        self,
        raw_gen: Generator[Tuple[str, Callable[[], pd.DataFrame]], None, None],
        keys: Optional[pd.DataFrame] = None,
        test_keys: Optional[pd.DataFrame] = None,
        multiprocess: bool = False,
    ) -> None:
        """
        Extract secondary metafeatures from the raw data set(s).

        Data sets are obtained via the `raw_gen` generator provided by DataSetParser
        subclasses. For each data set parser, all RawDataSetLambdaTransformers in the
        `featurizers` attribute is applied on the raw data set.

        The metafeatures extracted in this local context (stored in local variable `result`)
        will not have the same ordering as the `metafeatures` and
        `test_metafeatures` attribute in the DataSetParser subclasses, which
        prevents direct joins. To fix the ordering, `results` is rearranged by
        joining them on keys of the metafeatures and test_metafeatures
        (stored in `keys` and `test_keys` respectively).

        Arguments:
            raw_gen {Generator[Tuple[str, Callable[[], pd.DataFrame]], None, None]}
                -- A generator of raw data sets from which metafeatures are
                   extracted from.

        Keyword Arguments:
            keys {Optional[pd.DataFrame]}
                -- Keys from metafeatures to rearrange rows of `results`.
                   Expects a dataframe with the `attribute_name` and optionally
                   the `dataset` columns. (default: {None})
            test_keys {Optional[pd.DataFrame]}
                -- Keys from test metafeatures to rearrange rows of `result`.
                   Expects a dataframe with the `attribute_name` and optionally
                   the `dataset` columns. (default: {None})
            multiprocess {bool}
                -- If True, process the raw dataframes in parallel.
                   (default: {False})

        Returns:
            None
        """

        if multiprocess:
            with Pool(cpu_count() - 1) as p:
                func = partial(
                    _featurize_raw_dataframe, featurizers=self.featurizers
                )

                # Process in parallel
                results = list(
                    tqdm(
                        p.imap(func, raw_gen),
                        desc="Analyzing raw dataset (parallel)",
                        leave=False,
                    )
                )
        else:
            results = []
            # Loops through raw datasets provided by `raw_gen`
            for dataset, df in tqdm(
                raw_gen, desc="Analyzing raw dataset", leave=False
            ):
                results.append(
                    _featurize_raw_dataframe((dataset, df), self.featurizers)
                )

        # Combine all intermediary results into a master dataframe
        results = pd.concat(results, axis=0, ignore_index=True)

        # Left joins (test_key OR key) with `results`
        if keys is not None:
            self.sec_metafeatures = self.__map_features_to_feature_columns(
                keys, results
            )

        if test_keys is not None:
            self.sec_test_metafeatures = self.__map_features_to_feature_columns(
                test_keys, results
            )

        # Save precomputed features to skip
        # this slow calculation step in the future
        if self.save_dir:
            self.__serialize_features()

    def __map_features_to_feature_columns(
        self, keys: pd.DataFrame, results: List[pd.DataFrame]
    ) -> pd.Series:
        """
        Map newly created features to appropriate rows in the (test) metafeatures.

        Essentially this function selects and "reorders" the features.

        Arguments:
            keys {pd.DataFrame}
                -- Metafeatures containing `attribute_name` and optionally `dataset`.
            results {List[pd.DataFrame]}
                -- Test meatfeatures containing `attribute_name` and optionally `dataset.

        Raises:
            AssertionError -- When there is an imperfect mapping between features to data set.

        Returns:
            pd.Series -- Mapped features in correct order.
        """
        merged = pd.merge(
            keys,
            results,
            how="left",
            on=(
                ("dataset", "attribute_name")
                if "dataset" in keys
                else ("attribute_name",)
            ),
        )

        sample_cols = [
            col
            for col in merged.columns
            if "sample" in col and "samples" not in col
        ]

        if merged.drop(sample_cols, axis=1).isnull().any(axis=None):
            raise AssertionError(
                "Imperfect mapping from features to data set detected."
            )

        return merged[[f.method for f in self.featurizers]]

    def __get_feature_names(self) -> None:
        self.sec_feature_names = [
            name for f in self.featurizers for name in f.sec_feature_names
        ]

    def __serialize_features(self) -> None:
        obj = {
            "raw_features": [f.method for f in self.featurizers],
            "metafeatures": self.sec_metafeatures,
            "test_metafeatures": self.sec_test_metafeatures,
            "n_rows": (
                len(self.sec_metafeatures)
                if self.sec_metafeatures is not None
                else None,
                len(self.sec_test_metafeatures)
                if self.sec_test_metafeatures is not None
                else None,
            ),
        }
        io.to_pickle(obj, self.save_dir)

    def __is_same_file(self) -> bool:
        """
        Determine is file in `save_dir` should be fast-loaded.

        `save_dir` should only be loaded if:
            -- `n_rows` instance agree with `n_rows` in `save_dir`
            -- Number of RawDataSetLambdaTransformers agree with number of
               raw_features in `save_dir`
            -- `method` attribute of RawDataSetLambdaTransformers match
        """
        data = io.from_pickle(self.save_dir)

        # Check if length of secondary metafeatures match
        if data["n_rows"] != self.n_rows:
            return False

        # Check if the number of raw featurizers matches
        if len(data["raw_features"]) != len(self.featurizers):
            return False

        # Check if name of methods match
        method_names_all_match = all(
            [
                name == method
                for name, method in zip(
                    data["raw_features"], (f.method for f in self.featurizers)
                )
            ]
        )
        if not method_names_all_match:
            return False

        return True


class RawDataSetLambdaTransformerBuilder:
    """Builder class for RawDataSetLambdaTransformer."""

    def __call__(
        self,
        method: str,
        callable_: Callable[[pd.DataFrame], pd.Series],
        normalizable: bool,
        **_ignored,
    ):
        """Build a RawDataSetLambdaTransformer based on supplied keyword arguments."""
        return RawDataSetLambdaTransformer(
            method=method, callable_=callable_, normalizable=normalizable
        )


class RawDataSetFeaturizerViaLambdaBuilder:
    """Builder class for RawDataSetFeaturizerViaLambda."""

    def __call__(
        self,
        featurizers: List[Type[RawDataSetLambdaTransformer]],
        fast_load: bool = False,
        n_rows: Optional[Tuple[Optional[int], Optional[int]]] = None,
        save_dir: Optional[Path] = None,
        **_ignored,
    ):
        """Build a RawDataSetFeaturizerViaLambda based on supplied keyword arguments."""
        return RawDataSetFeaturizerViaLambda(
            featurizers=featurizers,
            fast_load=fast_load,
            n_rows=n_rows,
            save_dir=save_dir,
        )
