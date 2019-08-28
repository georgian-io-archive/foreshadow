"""A serializable form of sklearn pipelines."""

import inspect
import re

import six  # noqa: F401
from sklearn.base import clone  # noqa: F401
from sklearn.pipeline import Pipeline, _fit_transform_one  # noqa: F401
from sklearn.utils.validation import check_memory  # noqa: F401

from .parallelprocessor import ParallelProcessor  # noqa: F401
from .serializers import PipelineSerializerMixin


# Above imports used in runtime override.
# We need F401 as flake will not see us using the imports in exec'd code.


class SerializablePipeline(Pipeline, PipelineSerializerMixin):
    """sklearn.pipeline.Pipeline that uses PipelineSerializerMixin."""

    pass


def source(o):
    """Get source code of object, o.

    Args:
        o: Object to get source code of.

    Returns:
        source code as string.

    """
    s = inspect.getsource(o).split("\n")
    indent = len(s[0]) - len(s[0].lstrip())
    return "\n".join(i[indent:] for i in s)


_fit_source = source(Pipeline._fit)
_fit_source = re.sub(
    r"(for.+)(enumerate.+):", r"\1" + "enumerate(self.steps):", _fit_source
)
code = """#
            try:
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name]
                )
            except ValueError:  # single input required
                # ---------- THIS IS ONE CHANGE FOR SingleInputPipeline
                # Modifying the n+1 step to create a new transformer for
                # each column outputted by this step.
                # Only do this if its not the last step, because then
                # the next pipeline will have to handle the outputs of
                # this method. If its the last step, then we will just
                # output the DataFrame (which has more than one column)
                # and the next SmartTransformer will have to handle it
                # as its input.
                columns = Xt.columns
                transformer.transformer = None  # in case this is Smart and
                # resolved.
                transformer.should_resolve = True
                transformer = ParallelProcessor(
                        [
                            [
                                "dynamic_single_input_col_%d" % i,
                                clone(
                                    transformer
                                ),  # need separate instances
                                [columns[i]],
                            ]
                            for i in range(len(columns))
                        ],
                        collapse_index=True,
                    )
                if hasattr(memory, "cachedir") and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name]
                )
                self.steps[step_idx] = (name, fitted_transformer)"""
_fit_source = re.sub(
    r"(?sm)Xt, fitted_transformer.+fitted_transformer\)", code, _fit_source
)  # modifying fit portion
exec(compile(_fit_source, "<string>", "exec"))  # will compile to _fit
# I highly don't recommend using exec to do this, but seeing as we cannot
# copy scikit-learn code and super() won't work, this seems to be the only way.
# Debugging this when it fails will be an absolute pain: it will be spooky
# and mysterious. I recommend typing the code (rather than exec) if errors
# are encountered.


class DynamicPipeline(Pipeline, PipelineSerializerMixin):
    """Dynamically routes multiple outputs to separate Transformers."""

    # TODO replace with thorough dynamic pipeline that handles all use cases
    #  and is based off defined inputs/outputs for each transformer.
    def _fit(self, X, y=None, **fit_params):  # copied super method
        """Fit and then transform data.

        Fit all the transforms one after the other and transform the
        data. Has no final_estimator. If fitting errors out, it tries giving a
        single column to each SmartTransformer.

        Args:
            X (iterable): Training data. Must fulfill input requirements of
                first step of the pipeline.
            y (iterable, default=None): Training targets. Must fulfill label
                requirements for all steps of the pipeline.
            **fit_params (dict of string -> object): Parameters passed to the
                `fit` method of each step, where each parameter name is
                prefixed such that parameter `p` for step `s` has key
                `s__p`.

        Returns:
            Transformed inputs.

        """
        return _fit(self, X, y, **fit_params)  # noqa: F821

    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transforms one after the other and transform the
        data. Has no final_estimator.

        Args:
            X (iterable): Training data. Must fulfill input requirements of
                first step of the pipeline.
            y (iterable, default=None): Training targets. Must fulfill label
                requirements for all steps of the pipeline.
            **fit_params (dict of string -> object): Parameters passed to the
                `fit` method of each step, where each parameter name is
                prefixed such that parameter `p` for step `s` has key
                `s__p`.

        Returns:
            self : Pipeline, this estimator

        """
        self._fit(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.

        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Args:
            X (iterable): Training data. Must fulfill input requirements of
                first step of the pipeline.
            y (iterable, default=None): Training targets. Must fulfill label
                requirements for all steps of the pipeline.
            **fit_params (dict of string -> object): Parameters passed to
                the `fit` method of each step, where each parameter name
                is prefixed such that parameter `p` for step `s` has key
                `s__p`.

        Returns:
            Xt : array-like, shape = [n_samples, n_transformed_features] of
            Transformed samples

        """
        Xt, fit_params = self._fit(X, y, **fit_params)
        return Xt
