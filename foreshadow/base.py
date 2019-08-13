from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
import patchy

_set_params = BaseEstimator.set_params
patchy.patch(_set_params,
             """@@ -30,6 +30,6 @@
             setattr(self, key, value)
 
     for key, sub_params in nested_params.items():
-        valid_params[key].set_params(**sub_params)
+        getattr(self, key).set_params(**sub_params)
 
     return self
             """)


BaseEstimator.set_params = _set_params
TransformerMixin = TransformerMixin
