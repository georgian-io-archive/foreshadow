import json
import pickle
import inspect


def _resolve_pipeline(pipeline_json):
    """Deserializes pipeline from JSON into sklearn Pipeline object.

    Args:
        pipeline_json: list of form :code:`[cls, name, {**params}]`

    Returns:
        :obj:`Pipeline <sklearn.pipeline.Pipeline>`: Pipeline based on JSON

    Raises:
        KeyError: Malformed transformer while deserializing pipeline
        ValueError: Cannot import a defined transformer
        ValueError: Invalid parameters passed into transformer

    """
    pipe = []

    module_internals = __import__(
        "transformers.internals", globals(), locals(), ["object"], 1
    )
    module_externals = __import__(
        "transformers.externals", globals(), locals(), ["object"], 1
    )
    module_smart = __import__(
        "transformers.smart", globals(), locals(), ["object"], 1
    )

    for trans in pipeline_json:
        try:
            clsname = trans["transformer"]
            name = trans["name"]
            params = trans.get("parameters", {})

        except KeyError:
            raise KeyError(
                "Malformed transformer {} correct syntax is"
                '["transformer": cls, "name": name, "pipeline": '
                "{{**params}}]".format(trans)
            )

        try:
            search_module = (
                module_internals
                if hasattr(module_internals, clsname)
                else (
                    module_externals
                    if hasattr(module_externals, clsname)
                    else module_smart
                )
            )

            cls = getattr(search_module, clsname)

        except Exception:
            raise ValueError(
                "Could not import defined transformer {}".format(clsname)
            )

        try:
            pipe.append((name, cls(**params)))
        except TypeError:
            raise ValueError(
                "Params {} invalid for transfomer {}".format(
                    params, cls.__name__
                )
            )

    if len(pipe) == 0:
        return Pipeline([("null", None)])

    return Pipeline(pipe)


def deserialize(ser, method="json", **kwargs):
    """Deserialize data as specified.

    Args:
        method (str): A choice between `json`, `pickle`, and `joblib` to 
            serialize a string.
        name (str): Name of the transformer to serialize. Note that this
            resolves to 'var' if defined inline.
        **kwargs: The keyword arguments to pass to the 
            :meth:`sklearn.base.BaseEstimator.get_params` command.

    Returns:
        obj: The appropriate deserialized transformer.

    """

    pass


def _retrieve_name(var):
        """Gets the name of defined var.

        Source: https://stackoverflow.com/a/40536047

        Args:
            var: python object

        Returns:
            str: Instance name

        """
        for fi in reversed(inspect.stack()):
            names = [
                var_name
                for var_name, var_val in fi.frame.f_locals.items() 
                if var_val is var
            ]
            if len(names) > 0:
                return names[0]


class SerializerMixin(object):
    """Mixin class for all transformers to be saved to disk"""

    def __get_json(self, **kwargs):
        return self.get_params(**kwargs)

    def __get_pickle(self, **kwargs):
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    def serialize(self, method="json", name=None, **kwargs):
        """Serialize data as specified.

        Either freezes a transformer using pickle based serialization or
        through the use of json serialization. The former saves internal state
        of the transformer whereas the latter only saves the initilized
        configurations of the transformer. (ie it might need to be re-fit)

        Args:
            method (str): A choice between `json` and `pickle` to serialize a
                string.
            name (str): Name of the transformer to serialize. Note that this
                resolves to 'var' if defined inline.
            **kwargs: The keyword arguments (deep=True / False) to pass to the 
                :meth:`sklearn.base.BaseEstimator.get_params` command.

        Returns:
            str: The appropriate string representaiton of the serialization.

        """
        options = {"json": self.__get_json, "pickle": self.__get_pickle}

        try:
            params = options.get(method)(**kwargs)
        except KeyError:
            raise ValueError(
                "Serialization method must be one of {}".format(options.keys())
            )

        return json.dumps({
            'transformer': self.__class__.__name__,
            'name': name if name is not None else _retrieve_name(self),
            'params': params
        })

    def deserialize(self, method="json", **kwargs):
        """Deserialize data as specified.

        Args:
            method (str): A choice between `json`, `pickle`, and `joblib` to 
                serialize a string.
            name (str): Name of the transformer to serialize. Note that this
                resolves to 'var' if defined inline.
            **kwargs: The keyword arguments to pass to the 
                :meth:`sklearn.base.BaseEstimator.get_params` command.

        Returns:
            str: The appropriate string representaiton of the serialization.

        """
        pass

    def to_disk(self):
        pass

    def from_disk(self):
        pass
