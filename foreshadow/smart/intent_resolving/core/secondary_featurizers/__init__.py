"""
Module for secondary featurizer.

Concrete secondary featurizer classes include:
    -- RawDataSetFeaturizerViaLambda
    -- MetaDataSetFeaturizerViaLambda
    -- NGramFeaturizer
    -- Chars2VecFeaturizer

This __init__.py file also contains defintions for the followig helper class:
    -- FeaturizerCurator
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from .. import heuristics as hr  # Required for processing train_configs
from ..factory import GenericFactory as FeaturizerBuilderFactory
from .base_featurizer import BaseFeaturizer
from .meta_data_set_featurizer_via_lambda import (
    MetaDataSetFeaturizerViaLambda,
    MetaDataSetFeaturizerViaLambdaBuilder,
)
from .ngram_featurizer import NGramFeaturizer, NGramFeaturizerBuilder
from .raw_data_set_featurizer_via_lambda import (
    RawDataSetFeaturizerViaLambda,
    RawDataSetFeaturizerViaLambdaBuilder,
    RawDataSetLambdaTransformer,
    RawDataSetLambdaTransformerBuilder,
)


class FeaturizerCurator:
    """Class to build a list of secondary featurizers."""

    @staticmethod
    def from_cli(
        args: argparse.Namespace,
        n_rows: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> List[BaseFeaturizer]:
        """
        Build a list of secondary featurizers.

        Arguments:
            args {argparse.Namespace} -- Parsed CLI arguments.
            n_rows {Optional[Tuple[Optional[int],Optional[int]]]}
                -- Number of rows in metafeatures and test_metafeatures.
                   Required if `fast_load`-ing RawDataSetFeaturizerViaLambda.

        Returns:
            List[Type[BaseFeaturizer]] -- A list of secondary featurizers.
        """
        featurizers = []

        # Build function featurizers
        if "function_featurizers" in args.config:
            featurizers.extend(
                FeaturizerCurator._build_function_featurizers(
                    args.config["function_featurizers"],
                    n_rows=n_rows,
                    raw_lambda_transformer_config=args.config[
                        "raw_lambda_transformer_config"
                    ],
                )
            )

        # Build text featurizers
        if args.command:
            args_as_dict = vars(args)
            args_as_dict["text_mode"] = args.command
            featurizers.append(
                FeaturizerCurator._build_text_featurizers(**args_as_dict)
            )

        return featurizers

    @staticmethod
    def from_config(
        func_config: List[dict], text_config: Optional[dict] = None
    ) -> List[BaseFeaturizer]:
        """
        Build a list of secondary featurizers from a config list.

        Arguments:
            func_config {List[dict]} -- Configuration for each function featurizer.
            text_config {Optional[dict]} -- If provided, perform text featurization.

        Raises:
            ValueError -- `config` does not contain 'function_featurizers' key.

        Returns:
            List[BaseFeaturizer] -- A list of secondary fucntion featurizers.
        """
        featurizers = FeaturizerCurator._build_function_featurizers(
            func_config
        )

        # Perform text featurization
        if text_config is not None:
            featurizers.append(
                FeaturizerCurator._build_text_featurizers(**text_config)
            )

        return featurizers

    @staticmethod
    def _build_function_featurizers(
        function_featurizers_config: List[dict],
        n_rows: Optional[Tuple[Optional[int], Optional[int]]] = None,
        raw_lambda_transformer_config: dict = {},
    ) -> List[BaseFeaturizer]:

        featurizers = []
        raw_lambda_transformers = []

        # Create and append either a
        # `RawDataSetLambdaTransformer` or a `MetaDataSetFeaturizerViaLambda` instance
        # based the each featurizer's `on_raw` config parameter
        for featurizer_config in function_featurizers_config:
            featurizer_config["callable_"] = eval(
                featurizer_config["callable_"]
            )
            if featurizer_config["on_raw"]:
                raw_lambda_transformers.append(
                    factory.create(
                        "raw_lambda_transformer", **featurizer_config
                    )
                )
            else:
                featurizers.append(
                    factory.create("meta_numerical", **featurizer_config)
                )

        # Create and append a `RawDataSetFeaturizerViaLambda` instance
        if raw_lambda_transformers:
            featurizers.append(
                factory.create(
                    "raw_data_set_featurizer_via_lambda",
                    featurizers=raw_lambda_transformers,
                    **raw_lambda_transformer_config,
                    **{"n_rows": n_rows}
                )
            )

        return featurizers

    @staticmethod
    def _build_text_featurizers(**kwargs) -> BaseFeaturizer:
        return factory.create(kwargs["text_mode"], **kwargs)


# Register builders to featurizer factory
factory = FeaturizerBuilderFactory()
factory.register_builders(
    "raw_lambda_transformer", RawDataSetLambdaTransformerBuilder()
)
factory.register_builders(
    "raw_data_set_featurizer_via_lambda",
    RawDataSetFeaturizerViaLambdaBuilder(),
)
factory.register_builders(
    "meta_numerical", MetaDataSetFeaturizerViaLambdaBuilder()
)
factory.register_builders("ngram", NGramFeaturizerBuilder())
