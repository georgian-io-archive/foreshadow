"""Module containing the core IntentResolver logic to be used in production."""
from . import heuristics, io
from .data_set_parsers import DataFrameDataSetParser
from .intent_resolver import IntentResolver
from .secondary_featurizers import (
    FeaturizerCurator,
    factory as featurizer_factory,
)
