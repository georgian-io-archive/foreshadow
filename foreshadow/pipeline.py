"""A serializable form of sklearn pipelines."""

from sklearn.pipeline import Pipeline

from .serializers import PipelineSerializerMixin


# Above imports used in runtime override.
# We need F401 as flake will not see us using the imports in exec'd code.


class SerializablePipeline(Pipeline, PipelineSerializerMixin):
    """sklearn.pipeline.Pipeline that uses PipelineSerializerMixin."""

    pass
