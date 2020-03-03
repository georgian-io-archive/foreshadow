"""A serializable form of sklearn pipelines."""

from sklearn.pipeline import Pipeline


# Above imports used in runtime override.
# We need F401 as flake will not see us using the imports in exec'd code.

# TODO Remove and just use native Pipeline instead.
class SerializablePipeline(Pipeline):
    """sklearn.pipeline.Pipeline that uses PipelineSerializerMixin."""

    pass
