"""Smart Transformer and its helper methods."""

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin

from foreshadow.transformers.core.pipeline import SerializablePipeline
from foreshadow.transformers.core.wrapper import make_pandas_transformer
from foreshadow.utils import (
    check_df,
    get_transformer,
    is_transformer,
    is_wrapped,
)
from foreshadow.transformers.core.wrapper import _Empty


# TODO: Remove once _Empty is removed when DataCleaner is implemented
@make_pandas_transformer
class SmartTransformer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Abstract transformer class for meta transformer selection decisions.

    This class contains the logic necessary to determine a single transformer
    or pipeline object that should act in its place.

    Once in a pipeline this class can be continuously re-fit in order to adapt
    to different data sets.

    Contains a function _get_tranformer that must be overridden by an
    implementing class that returns a scikit-learn transformer object to be
    used.

    Used and implements itself identically to a transformer.

    Attributes:
        override: A scikit-learn transformer that can be optionally provided
            to override internals logic. This takes top priority of all the
            setting.
        should_resolve: Whether or not the SmartTransformer will resolve
            the concrete transformer determination on each fit. This flag will
            set to `False` after the first fit. If force_reresolve is set, this
            will be ignored.
        force_reresolve: Forces re-resolve on each fit. If override is set it
            takes top priority.
        **kwargs: If overriding the transformer, these kwargs passed downstream
            to the overridden transformer

    """

    def __init__(self,
                 y_var=False,
                 override=None,
                 should_resolve=True,
                 force_reresolve=False,
                 # column_sharer=None,
                 **kwargs,
                 ):
        self.kwargs = kwargs
        # self.column_sharer=column_sharer
        # TODO will need to add the above when this is no longer wrapped
        self.y_var = y_var
        self.transformer = None
        self.should_resolve = should_resolve
        self.force_reresolve = force_reresolve
        # Needs to be declared last as this overrides the resolve parameters
        self.override = override

    @property
    def transformer(self):
        """Get the selected transformer from the SmartTransformer.

        Returns:
            object: An instance of a concrete transformer.

        """
        return self._transformer

    @transformer.setter
    def transformer(self, value):
        """Validate transformer initialization.

        Args:
            value (object): The selected transformer that SmartTransformer
                should use.

        Raises:
            ValueError: If input is neither a valid foreshadow wrapped
                transformer, scikit-learn Pipeline, scikit-learn FeatureUnion,
                nor None.

        """
        # Check transformer type
        is_trans = is_transformer(value) and is_wrapped(value)
        is_pipe = isinstance(value, SerializablePipeline)
        is_none = value is None
        is_empty = isinstance(value, _Empty)
        checks = [is_trans, is_pipe, is_none, is_empty]
        # Check the transformer inheritance status
        if not any(checks):
            raise ValueError(
                "{} is neither a scikit-learn Pipeline, FeatureUnion, a "
                "wrapped foreshadow transformer, nor None.".format(value)
            )

        self._transformer = value

    @property
    def override(self):
        """Get the override parameter.

        Returns:
            str: The name of the override transformer class.

        """
        return self._override

    @override.setter
    def override(self, value):
        """Set the override parameter using a string class name.

        Args:
            value (str): The name of the desired transformer.

        """
        # self.name and self.keep_columns are injected as a result of pandas
        # wrapping. Try to resolve the transformer, otherwise error out.
        if value is not None:
            self.transformer = get_transformer(value)(
                name=self.name, keep_columns=self.keep_columns, **self.kwargs
            )
            self.transformer.name = self.name
            self.transformer.keep_columns = self.keep_columns
            self.should_resolve = False
            self.force_reresolve = False

        self._override = value

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Note: self.name and self.keep_columns are provided by the wrapping
            method

        Args:
            deep (bool): If True, will return the parameters for this estimator
                and contained sub-objects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        return {
            "y_var": self.y_var,
            "override": self.override,
            "name": self.name,
            "keep_columns": self.keep_columns,
            "column_sharer": self.column_sharer,
            **(
                self.transformer.get_params(deep=deep)
                if self.transformer is not None and deep
                else {}
            ),
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with :meth:`get_params()`.

        Args:
            **params (dict): any valid parameter of this estimator

        """
        self.name = params.pop("name", self.name)
        self.keep_columns = params.pop("keep_columns", self.keep_columns)
        self.y_var = params.pop("y_var", self.y_var)

        # Calls to override auto set the transformer instance
        self.override = params.pop("override", self.override)

        if self.transformer is not None:
            valid_params = {
                k.partition("__")[2]: v
                for k, v in params.items()
                if k.split("__")[0] == "transformer"
            }
            self.transformer.set_params(**valid_params)

    @abstractmethod
    def pick_transformer(self, X, y=None, **fit_params):
        """Pick the correct transformer object for implementations.

        Args:
            X (:obj:`pandas.DataFrame`): Input X data
            y (:obj: 'pandas.DataFrame'): labels Y for data
            **fit_params (dict): Parameters to apply to transformers when
                fitting

        """
        pass  # pragma: no cover

    def resolve(self, X, y=None, **fit_params):
        """Verify transformers have the necessary methods and attributes.

        Args:
            X: input observations
            y: input labels
            **fit_params: params to fit

        """
        # If override is passed in or set, all types of resolves are turned
        # off.
        # Otherwise, force_reresolve will always resolve on each fit.

        # If force_reresolve is set, always re-resolve
        if self.force_reresolve:
            self.should_resolve = True

        # Only resolve if transformer is not set or re-resolve is requested.
        if self.should_resolve:
            self.transformer = self.pick_transformer(X, y, **fit_params)
            if getattr(self.transformer, 'name', None) is None:
                self.transformer.name = self.name
            self.transformer.keep_columns = self.keep_columns

        # reset should_resolve
        self.should_resolve = False

    def transform(self, X):
        """See base class.

        Args:
            X: transform

        Returns:
            transformed X using selected best transformer.

        """
        X = check_df(X)
        self.resolve(X)
        return self.transformer.transform(X)

    def fit(self, X, y=None, **fit_params):
        """See base class.

        This class returns self, not self.transformer.fit, which would
        return the aggregated transformers self because then chains such as
        SmartTransformer().fit().transform() would only call the underlying
        transformer's fit. In the case that Smart is Wrapped, this changes
        the way columns are named.

        Args:
            X: see base class
            y: see base class
            **fit_params: see base class

        Returns:
            see base class

        """
        X = check_df(X)
        y = check_df(y, ignore_none=True)
        self.resolve(X, y, **fit_params)
        self.transformer.full_df = fit_params.pop("full_df", None)
        self.transformer.fit(X, y, **fit_params)
        return self  # .transformer.fit(X, y, **fit_params)
        # This should not return the self.transformer.fit as that will
        # cause fit_transforms, which call .fit().transform() to fail when
        # using our wrapper for transformers; TL;DR, it misses the call to
        # this class's transform.

    def inverse_transform(self, X):
        """Invert transform if possible.

        Args:
            X: transformed input observations using selected best transformer

        Returns:
            original input observations

        """
        X = check_df(X)
        self.resolve(X)
        return self.transformer.inverse_transform(X)
