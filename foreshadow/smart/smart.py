"""Smart Transformer and its helper methods."""

from abc import ABCMeta, abstractmethod

from sklearn.pipeline import Pipeline

from foreshadow.base import BaseEstimator, TransformerMixin
from foreshadow.logging import logging
from foreshadow.utils import (
    UserOverrideMixin,
    check_df,
    get_transformer,
    is_transformer,
    is_wrapped,
)


class SmartTransformer(
    BaseEstimator, TransformerMixin, UserOverrideMixin, metaclass=ABCMeta
):
    """Abstract transformer class for meta transformer selection decisions.

    This class contains the logic necessary to determine a single transformer
    or pipeline object that should act in its place.

    Once in a pipeline this class can be continuously re-fit in order to adapt
    to different data sets.

    Contains a function pick_tranformer that must be overridden by an
    implementing class that returns a scikit-learn transformer object to be
    used.

    Note that by default the return value of pick_tranformer has multiple
    validation checks make sure that it will work with the rest of the system.
    To simply check that the return value is any "transformer", set the
    `validate_wrapped` class attribute in subclasses.

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

    validate_wrapped = True

    def __init__(
        self,
        y_var=False,
        transformer=None,
        should_resolve=True,
        force_reresolve=False,
        cache_manager=None,
        name=None,
        keep_columns=False,
        check_wrapped=True,
        **kwargs,
    ):
        self.name = name
        self.keep_columns = keep_columns
        self.kwargs = kwargs
        self.cache_manager = cache_manager
        # TODO will need to add the above when this is no longer wrapped
        self.y_var = y_var
        self.should_resolve = should_resolve
        self.force_reresolve = force_reresolve
        # Needs to be declared last as this overrides the resolve parameters
        self.transformer = transformer
        self.check_wrapped = check_wrapped

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
        if isinstance(value, str):
            value = get_transformer(value)(**self.kwargs)
            self.unset_resolve()
        # Check transformer type
        is_trans = is_transformer(value)
        trans_wrapped = (
            is_wrapped(value) if getattr(self, "check_wrapped", True) else True
        )
        # True by default passes this check if we don't want it.
        is_pipe = isinstance(value, Pipeline)
        is_none = value is None
        checks = [is_trans, is_pipe, is_none, trans_wrapped]
        # Check the transformer inheritance status
        if not any(checks):
            logging.error(
                "transformer: {} failed checks: {}".format(value, checks)
            )
            raise ValueError(
                "{} is neither a scikit-learn Pipeline, FeatureUnion, a "
                "wrapped foreshadow transformer, nor None.".format(value)
            )

        self._transformer = value

    def unset_resolve(self):
        """Unset resolving for all passes."""
        self.should_resolve = False
        self.force_reresolve = False

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

    def _has_fitted(self):
        """Check if the SmartTransformer has resolved or not.

        Returns:
            bool: has fitted flag

        """
        return self.should_resolve is False

    def resolve(self, X, y=None, **fit_params):
        """Verify transformers have the necessary methods and attributes.

        Args:
            X: input observations
            y: input labels
            **fit_params: params to fit

        """
        self.force_reresolve = (
            self.force_reresolve
            or self.should_force_reresolve_based_on_override(X)
        )

        if self.force_reresolve:
            self.should_resolve = True

        # Only resolve if transformer is not set or re-resolve is requested.
        if self.should_resolve:
            self.transformer = self.pick_transformer(
                X.copy() if X is not None else X,
                y.copy() if y is not None else y,
                **fit_params,
            )
            if getattr(self.transformer, "name", None) is None:
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

        Raises:
            ValueError: Transformer should be fitted first.

        """
        X = check_df(X)
        # Why do we need to resolve twice? If the transformer is not set,
        # throw an exception since we should call fit before transform
        # self.resolve(X)
        if self.transformer is None:
            raise ValueError(
                "The transformer has not been fitted. Please "
                "call fit first."
            )
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
        return self
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

    @classmethod
    def _get_param_names(cls):
        """Get iteratively __init__ params for all classes until PreparerStep.

        Returns:
            params for all parents up to and including PreparerStep.
            Includes the calling classes params.

        """
        params = super()._get_param_names()
        while cls.__name__ != SmartTransformer.__name__:
            cls = cls.__mro__[1]
            params += cls._get_param_names()
        return params
