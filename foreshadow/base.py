"""Foreshadow version of sklearn.base.py."""
import patchy
from sklearn.base import TransformerMixin  # noqa: F401
from sklearn.base import BaseEstimator


_set_params = BaseEstimator.set_params
patchy.patch(
    _set_params,
    """@@ -30,6 +30,9 @@
             setattr(self, key, value)

     for key, sub_params in nested_params.items():
-        valid_params[key].set_params(**sub_params)
+        try:
+            getattr(self, key).set_params(**sub_params)
+        except AttributeError:  # for Pipelines
+            valid_params[key].set_params(**sub_params)

     return self
             """,
)
"""sklearn.base.BaseEstiamtor uses the valid_params to set the params.

In our use cases, we often modify both an object and its params. In this case,
the setattr(self, key, value) will change this object (key will refer to its
attribute on the parent object, value to the object itself), but the
valid_params[key] will have a reference to the old aggregate object,
not setting the params on the new object. This is a big issue when we try to
simultaneously change both an object and its params. For instance,
see smart where we set both a transformer and that transformer's params.

In the case of Smart,
where Smart.transformer is a Transformer object, we would see this:

smart = Smart()
smart.transformer = StandardScaler()

smart.set_params({'transformer' BoxCox(), 'transformer__param': some_value})

First, we get the valid params for this object (smart).
valid_params = self.get_params()
# valid_params['transformer'] == StandardScaler

get_params does some checking on the params being set.
Now, get_params will set the transformer instance first, before its nested
params, which is desired.

setattr(self, 'transformer', BoxCox())

# Note, valid_params['transformer'] is still StandardScaler.

Now, we set the nested params for the smart.transformer object
({'transformer__param': some_value})

We do this in the nested_params section, which will use the previously
acquired valid_params.
valid_params['transformer'].set_params({'transformer__param': some_value})

This would in fact be StandardScaler, NOT BoxCox!.
This is why we do getattr to get the BoxCox which would have been previously
set by the setattr call above.


we default back to valid_params[key] when this fails as we are dealing with
a Pipeline object which works differently.
"""


BaseEstimator.set_params = _set_params
