"""Module containing the core IntentResolver logic to be used in production."""
from . import heuristics, io
from .data_set_parser import DataFrameDataSetParser
from .intent_resolver import IntentResolver
from .secondary_featurizer import (
    FeaturizerCurator,
    factory as featurizer_factory,
)
