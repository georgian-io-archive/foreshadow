"""A serializable form of sklearn pipelines."""

from sklearn.pipeline import Pipeline

from foreshadow.core import PipelineSerializerMixin


class SerializablePipeline(Pipeline, PipelineSerializerMixin):
    """sklearn.pipeline.Pipeline that uses PipelineSerializerMixin."""

    pass
