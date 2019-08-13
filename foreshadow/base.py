"""Foreshadow version of sklearn.base.py."""
import patchy
from sklearn.base import BaseEstimator, TransformerMixin


_set_params = BaseEstimator.set_params
patchy.patch(
    _set_params,
    """@@ -30,6 +30,6 @@
             setattr(self, key, value)

     for key, sub_params in nested_params.items():
-        valid_params[key].set_params(**sub_params)
+        getattr(self, key).set_params(**sub_params)

     return self
             """,
)
"""sklearn.base.BaseEstiamtor uses the valid_params to set the params.

In our use cases, we often modify both an objet and its params. In this case,
the setattr(self, key, value) will change this object, but the
valid_params[key] will have a reference to the old object, not setting the
params on the new object. This is a big issue when we try to simultaneously
change both an object and its params simultaneously. For instsance,
see smart where we set both a transformer and that transformer's params.
"""


BaseEstimator.set_params = _set_params
TransformerMixin = TransformerMixin
