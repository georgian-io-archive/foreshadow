"""A serializable form of sklearn pipelines."""
import six
from sklearn.pipeline import Pipeline

from foreshadow.core import PipelineSerializerMixin
from foreshadow.transformers.core import ParallelProcessor
from foreshadow.utils.validation import check_df
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_memory
from sklearn.base import clone
from sklearn.pipeline import _fit_transform_one


class SerializablePipeline(Pipeline, PipelineSerializerMixin):
    """sklearn.pipeline.Pipeline that uses PipelineSerializerMixin."""

    pass


class SingleInputPipeline(Pipeline):
    def _fit(self, X, y=None, **fit_params):  # copied super method
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        # A CHANGE FOR SingleInputPipeline, steps has no final_estimator
        for step_idx, (name, transformer) in enumerate(self.steps):
            if transformer is None:
                pass
            else:
                if hasattr(memory, 'cachedir') and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name])
                # ------------ THIS IS ONE CHANGE FOR SingleInputPipeline
                is_single_column = True
                try:
                    check_df(Xt, single_or_empty=True)
                except:
                    columns = Xt.columns
                    is_single_column = False
                if not is_single_column and step_idx < len(self.steps) - 1:
                    # ---------- THIS IS ONE CHANGE FOR SingleInputPipeline
                    # Modifying the n+1 step to create a new transformer for
                    # each column outputted by this step.
                    # Only do this if its not the last step, because then
                    # the next pipeline will have to handle the outputs of
                    # this method. If its the last step, then we will just
                    # output the DataFrame (which has more than one column)
                    # and the next SmartTransformer will have to handle it
                    # as its input.
                    next_step, next_trans = self.steps[step_idx+1]
                    self.steps[step_idx+1] = (
                        '%s' % next_step,
                        ParallelProcessor(
                            [['dynamic_single_input_col_%d' % i,
                              clone(next_trans),  # need separate instances
                              [columns[i]]] for i in range(len(columns))]
                        )
                    )
                # ------------------------------------------------------------
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)
        return Xt

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data. Has no final_estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        self._fit(X, y, **fit_params)
        return self


class TransformersPipeline(Pipeline):
    def _fit(self, X, y=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for step_idx, (name, transformer) in enumerate(self.steps):
            if transformer is None:
                pass
            else:
                if hasattr(memory, 'cachedir') and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name])
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)
        return Xt

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, fit_params = self._fit(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator

        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        """
        last_step = self._final_estimator
        Xt, fit_params = self._fit(X, y, **fit_params)
        if hasattr(last_step, 'fit_transform'):
            return last_step.fit_transform(Xt, y, **fit_params)
        elif last_step is None:
            return Xt
        else:
            return last_step.fit(Xt, y, **fit_params).transform(Xt)