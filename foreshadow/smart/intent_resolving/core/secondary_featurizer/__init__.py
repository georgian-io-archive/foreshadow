"""
Module for secondary featurizer.

Concrete secondary featurizer classes include:
    -- RawDataSetFeaturizerViaLambda
    -- MetaDataSetFeaturizerViaLambda
    -- NGramFeaturizer
    -- Chars2VecFeaturizer
    -- FastTextFeaturizer

This __init__.py file also contains defintions for the followig helper class:
    -- FeaturizerCurator
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import pandas as pd

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
    """Class to build a list of secondary featurizers from parsed CLI args."""

    @staticmethod
    def curate(
        args: argparse.Namespace,
        n_rows: Optional[Tuple[Optional[int], Optional[int]]],
    ) -> List[Type[BaseFeaturizer]]:
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

        # Build numerical featurizers
        if "numerical_featurizers" in args.config:
            raw_lambda_transformers = []

            # Create and append either a
            # `RawDataSetLambdaTransformer` or a `MetaDataSetFeaturizerViaLambda` instance
            # based the each featurizer's `on_raw` config parameter
            for featurizer_config in args.config["numerical_featurizers"]:
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
                        **args.config["raw_lambda_transformer_config"],
                        **{"n_rows": n_rows}
                    )
                )

        # Build text featurizers
        if args.command:
            featurizers.append(factory.create(args.command, **vars(args)))

        return featurizers


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
