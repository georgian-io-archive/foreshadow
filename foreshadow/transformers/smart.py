"""Smart Transformers

Transformers here will be accessible through the namespace
foreshadow.transformers.smart OR simple foreshadow.transformers and will not be
wrapped or transformed. Only classes extending SmartTransformer should exist here.

"""

from .transformers import SmartTransformer
from .transformers import Imputer


class SimpleImputer(SmartTransformer):
    def _get_transformer(self, X, y=None, **fit_params):

        return Imputer()


class MultiImputer(SmartTransformer):
    def _get_transformer(self, X, y=None, **fit_params):

        return Imputer()
