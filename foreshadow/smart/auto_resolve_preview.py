import pandas as pd
from sklearn.datasets import load_breast_cancer

from foreshadow.intents import Categoric, Numeric, Text
from foreshadow.smart.intent_resolving.core import (
    IntentResolver as AutoIntentResolver,
)
from foreshadow.utils import get_transformer


_temporary_naming_conversion = {
    "Numerical": Numeric.__name__,
    "Categorical": Categoric.__name__,
    "Neither": "Neither",
}


def _temporary_naming_convert(auto_ml_intent_name):
    if auto_ml_intent_name in _temporary_naming_conversion:
        return _temporary_naming_conversion[auto_ml_intent_name]
    else:
        raise KeyError(
            "No such intent type {} exists.".format(auto_ml_intent_name)
        )


cancer = load_breast_cancer()
cancerX_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

resolver = AutoIntentResolver(cancerX_df)
result = resolver.predict()
print(result)
print(result[[0]])
print(result[[0]].values)
print(result[[0]].values[0])
class_name = result[[0]].values[0]
print(get_transformer(_temporary_naming_convert(class_name)))
