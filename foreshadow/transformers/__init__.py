from copy import deepcopy
import glob
import inspect
import os

import numpy as np
import pandas as pd

from copy import deepcopy
from functools import partialmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import (
    FeatureUnion,
    _fit_one_transformer,
    _fit_transform_one,
    _transform_one,
)
from sklearn.externals.joblib import Parallel, delayed


def get_classes():
    """Recursively searches transforms directory for classes"""

    files = glob.glob(os.path.dirname(__file__) + "/*.py")
    imports = [
        os.path.basename(f)[:-3]
        for f in files
        if os.path.isfile(f) and not f.endswith("__init__.py")
    ]
    modules = [__import__(i, globals(), locals(), ["object"], 1) for i in imports]
    classes = [
        c[1] for m in modules for c in inspect.getmembers(m) if inspect.isclass(c[1])
    ]

    return classes


def get_modules():
    """Recursively searches transforms directory for files containing classes
       extending sklearn.base.TransformerMixin
    """

    transformers = [
        cls
        for cls in get_classes()
        if issubclass(cls, TransformerMixin) and issubclass(cls, BaseEstimator)
    ]

    for t in transformers:
        copied_t = type(t.__name__, t.__bases__, dict(t.__dict__))
        globals()[copied_t.__name__] = wrap_transformer(copied_t)

    return len(transformers)


def wrap_transformer(transformer):
    """Wraps an sklearn transformer in functions allowing it to operate on a column basis
    """

    members = [
        m[1] for m in inspect.getmembers(transformer, predicate=inspect.isfunction)
    ]
    wrap_candidates = [
        m
        for m in members
        if "X" in inspect.getfullargspec(m).args and not m.__name__[0] == "_"
    ]

    for w in wrap_candidates:

        # Wrap public function calls
        setattr(transformer, w.__name__, partialmethod(pandas_wrapper, w))

    inspect.getfullargspec(transformer.__init__)
    # Wrap constructor
    setattr(
        transformer,
        "__init__",
        Sigcopy(transformer.__init__)(init_partial(transformer.__init__)),
    )

    return transformer

class Sigcopy(object):
    """
    copy_argspec is a signature modifying decorator.  Specifically, it copies
    the signature from `source_func` to the wrapper, and the wrapper will call
    the original function (which should be using *args, **kwds).  The argspec,
    docstring, and default values are copied from src_func, and __module__ and
    __dict__ from tgt_func.
    """
    def __init__(self, src_func):
        self.argspec = inspect.getfullargspec(src_func)
        self.src_doc = src_func.__doc__
        self.src_defaults = src_func.__defaults__

    def __call__(self, tgt_func):
        tgt_argspec = inspect.getfullargspec(tgt_func)
        need_self = False
        if tgt_argspec[0][0] == 'self':
            need_self = True
            
        name = tgt_func.__name__
        argspec = self.argspec
        argspec = argspec[0:3] + (tuple([s if type(s).__name__ != 'type' else None for s in argspec[3]]),) + argspec[4:]

        if argspec[0][0] == 'self':
            need_self = False
        if need_self:
            newargspec = (['self'] + argspec[0] + tgt_argspec[0][1:],) + argspec[1:4] + (tgt_argspec[4], tgt_argspec[5],) + argspec[6:]
        else:
            newargspec = (argspec[0] + tgt_argspec[0][1:],) + argspec[1:4] + (tgt_argspec[4], tgt_argspec[5],) + argspec[6:]
        sigcall = inspect.formatargspec(formatvalue=lambda val: "",
                *newargspec
                )[1:-1]
        signature = inspect.formatargspec(
                *newargspec
                )[1:-1]


        signature = signature.replace('*,', '')
        sigcall = sigcall.replace('*,', '')

        new_func = (
                'def _wrapper_(%(signature)s):\n' 
                '    return %(tgt_func)s(%(sigcall)s)' %
                {'signature':signature, 'tgt_func':'tgt_func', 'sigcall':sigcall}
                   )

        evaldict = {'tgt_func': tgt_func}
        exec(new_func, evaldict)
        wrapped = evaldict['_wrapper_']
        wrapped.__name__ = name
        wrapped.__doc__ = self.src_doc
        wrapped.__module__ = tgt_func.__module__
        wrapped.__dict__ = tgt_func.__dict__
        return wrapped


def init_partial(func):

    def transform_constructor(self, *args, keep_columns=False, name=None):

        self.name = name
        self.keep_columns = keep_columns
        func(self, *args[:-2])

    return transform_constructor


def pandas_wrapper(self, func, df, *args, **kwargs):
    """Wrapper function to replace public transformer functions.
    Selects columns from df and executes inner function only on columns.

    This expects that public functions within the sklearn transformer follow the sklearn
    standard. This includes the format func(X, y=None, *args, **kwards) and either a
    return self or return X

    :arg func: Original function to be wrapped
    :arg self: Transformer instance object
    :arg df: Input data (pd.Dataframe)

    :return output: Returns transformer object or pd.Dataframe
    """

    stack = inspect.stack()
    caller = None
    try:
        caller = stack[1][0].f_locals["self"].__class__
        if caller.__name__ == type(self).__name__:
            return func(self, df, *args, **kwargs)
    except:
        pass

    init_cols = [col for col in df]
    out = func(self, df, *args, **kwargs)

    # If output is numpy array (transform has occurred)
    if isinstance(out, np.ndarray):

        # Remove old columns if necessary
        if not self.keep_columns:
            df = df[[]]

        # Determine name of new columns
        if self.name:
            prefix = self.name
        else:
            prefix = type(self).__name__

        # Append new columns to data frame
        for i, col in enumerate(out.transpose().tolist()):
            kw = {"{}_{}_{}".format("_".join(init_cols), prefix, i): pd.Series(col)}
            df = df.assign(**kw)

        return df

    return out


class ParallelProcessor(FeatureUnion):
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None, collapse_index=False):

        self.collapse_index = collapse_index

        for i, item in enumerate(transformer_list):
            self._set_names(item)

        super(ParallelProcessor, self).__init__(
            transformer_list, n_jobs, transformer_weights
        )

    def _set_names(self, item):
        if hasattr(item[-1], 'name'):
            item[-1].name = item[0]
        if hasattr(item[-1], 'steps'):
            for step in item[-1].steps:
                self._set_names(step)
        if hasattr(item[-1], 'transformer_list'):
            for trans in item[-1].transformer_list:
                self._set_names(trans)

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [
            (name, cols, None if old is None else next(transformers))
            for name, cols, old in self.transformer_list
        ]

    def _validate_transformers(self):
        names, cols, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t is None:
                continue
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All estimators should implement fit and "
                    "transform. '%s' (type %s) doesn't" % (t, type(t))
                )

    def _iter(self):
        """Generate (name, est, weight) tuples excluding None transformers
        """
        get_weight = (self.transformer_weights or {}).get

        return (
            (name, cols, trans, get_weight(name))
            for name, cols, trans in self.transformer_list
            if trans is not None
        )

    def _get_other_cols(self, X):

        full = set(list(X))
        partial = set(list(self._slice_cols(X, [c for _, cols, _, _ in self._iter() for c in cols], drop_level=False)))

        return list(full - partial)

    def _slice_cols(self, X, cols, drop_level=True):
        origin = list(X)
        if len(origin) == 0:
            return X
        if len(cols) == 0:
            return X.drop(list(X), axis=1)
        if type(origin[0]) == tuple:
            origin, new = list(zip(*origin))
        else:
            return X[cols]

        def get(c, level):
            try:
                ret = X.xs(c, axis=1, level=level, drop_level=False)
                if drop_level:
                    ret.columns = ret.columns.droplevel()
                return ret
            except AttributeError as e:
                return X.xs(c, axis=1, level=level)
            except KeyError as e:
                return X.drop(list(X), axis=1)

        df = pd.concat([get(c.replace('$', ''), 'new')
                        if c[0] == '$'
                        else get(c, 'origin')
                        for c in cols
                        if c in origin or c.replace('$', '') in new], axis=1)
        return df

    def fit(self, X, y=None):
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()

        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, self._slice_cols(X, cols), y)
            for name, cols, trans, weight in self._iter()
        )

        self._update_transformer_list(transformers)

        return self

    def _pandas_transform_one(self, transformer, weight, X, cols):
        if len(cols) == 0:
            return X[[]]
        colname = cols[0]
        res = _transform_one(transformer, weight, X)
        res.columns = [[colname]*len(list(res)), list(res)]
        res.columns = res.columns.rename(['origin', 'new'])
        return res

    def _pandas_fit_transform_one(self, transformer, weight, X, y, cols, **fit_params):
        if len(cols) == 0:
            return X[[]], transformer
        colname = cols[0]
        res, t = _fit_transform_one(transformer, weight, X, y, **fit_params)
        res.columns = [[colname]*len(list(res)), list(res)]
        res.columns = res.columns.rename(['origin', 'new'])
        return res, t

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(self._pandas_transform_one)(trans, weight, self._slice_cols(X, cols), cols)
            for name, cols, trans, weight in self._iter()
        )
        Xs += (X[self._get_other_cols(X)],)

        if not Xs:
            # All transformers are None
            return X[[]]
        else:
            Xs = pd.concat(Xs, axis=1)

        if self.collapse_index:
            Xs.columns = Xs.columns.droplevel()
        return Xs

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(self._pandas_fit_transform_one)(trans, weight, self._slice_cols(X, cols), y, cols, **fit_params)
            for name, cols, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return X[[]]

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        Xs += (X[self._get_other_cols(X)],)
        Xs = pd.concat(Xs, axis=1)

        if self.collapse_index:
            Xs.columns = Xs.columns.droplevel()
        return Xs


class SmartTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, override=None):
        self.override = override
        self.transformer = None
        super(SmartTransformer, self).__init__()

    def _get_transformer(self, X, **fit_params):
        raise NotImplementedError(
            "WrappedTransformer _get_transformer was not implimented."
        )

    def _verify_transformer(self, X, **fit_params):

        if self.transformer:
            return

        self.transformer = self._get_transformer(X, **fit_params)

        if not self.transformer:
            raise AttributeError(
                "Invalid WrappedTransformer. Get transformer returns invalid object"
            )

        tf = getattr(self.transformer, "transform", None)
        fittf = getattr(self.transformer, "fit_transform", None)
        fit = getattr(self.transformer, "fit", None)

        if not (callable(tf) and callable(fittf) and callable(fit)):
            raise AttributeError(
                "Invalid WrappedTransformer. Get transformer returns invalid object"
            )

    def fit_transform(self, X, y=None, **fit_params):
        if self.override is not None:
            return self.override.fit_transform(X, y, **fit_params)
        self._verify_transformer(X, **fit_params)
        return self.transformer.fit_transform(X, y, **fit_params)

    def transform(self, X, **kwargs):
        if self.override is not None:
            return self.override.transform(X, **kwargs)
        return self.transformer.transform(X, **kwargs)

    def fit(self, X, y=None, **kwargs):
        if self.override is not None:
            return self.override.fit(X, y, **kwargs)
        self._verify_transformer(X)
        return self.transformer.fit(X, y, **kwargs)


# TODO: Determine verbosity of this process. Global?
print("Loading transformers....")
n = get_modules()
print("Loaded {} transformer plugins".format(n))
