"""Smart Transformers

Transformers here will be accessible through the namespace
foreshadow.transformers.smart OR simple foreshadow.transformers and will not be
wrapped or transformed. Only classes extending SmartTransformer should exist here.

"""

from .imputer import SimpleImputer, MultiImputer
