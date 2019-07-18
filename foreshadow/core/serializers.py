"""Transformer serialization utilities."""

import inspect
import json
import os
import pickle
import uuid

import yaml

from foreshadow.utils import get_cache_path, get_transformer


def _retrieve_name(var):
    """Get the name of defined var.

    Args:
        var: python object

    Returns:
        str: Instance name

    """
    # Source: https://stackoverflow.com/a/40536047
    for fi in reversed(inspect.stack()):
        names = [
            var_name
            for var_name, var_val in fi.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            return names[0]


class BaseTransformerSerializer:
    """The base transformer serialization class.

    Defines the basic option routing for serialize and deserialize operations.
    In order to add a deserialization option, subclass from the class, add the
    method's name to the `OPTIONS` list, and implement two versions of the
    method, a method_serialize & a method_deserialize.

    Use the `DEFAULT_OPTION` class attribute to set the default pathway method
    for serialization and deserialization. Use the `DEFAULT_KWARGS` class
    attribute to override the parent class's default.

    """

    OPTIONS = []
    DEFAULT_OPTION = None

    def serialize(self, method=None, **kwargs):
        """Specify the method routhing for a transformer serialization.

        Args:
            method (str): An option from the OPTIONS list defined in the class
                base
            **kwargs: Any options that need to be passed to a specific method.

        Returns:
            dict: A dictioanry representation of the transformer

        Raises:
            ValueError: If the method is not in `OPTIONS`

        """
        if method is None:
            method = self.DEFAULT_OPTION

        if method in self.OPTIONS:
            method_func = getattr(self, method + "_serialize")

            payload = method_func(**kwargs)
        else:
            raise ValueError(
                "Serialization method must be one of {}".format(self.OPTIONS)
            )

        return {"class": self.__class__.__name__, "method": method, **payload}

    @classmethod
    def deserialize(cls, data):
        """Specify the method routing for a transformer deserialization.

        Args:
            data (dict): The counterpart to serialize that has all the required
                args to build a transformer.

        Returns:
            object: The deserialized transformer

        Raises:
            ValueError: If the method is not in `OPTIONS`

        """
        method = data["method"]  # TODO: add malformed serialization error
        if method in cls.OPTIONS:
            method_func = getattr(cls, method + "_deserialize")
            return method_func(data)
        else:
            raise ValueError(
                "Deserialization method must be one of {}".format(cls.OPTIONS)
            )


class ConcreteSerializerMixin(BaseTransformerSerializer):
    """Mixin class that provides convenience serialization methods."""

    OPTIONS = ["dict", "inline", "disk"]
    DEFAULT_OPTION = "dict"

    def _pickle_cache_path(self, cache_path=None):
        """Get the pickle cache path of a transformer.

        Uses a generated UUID and the class name to come up with a unique
        filename.

        Args:
            cache_path (str, optional): override the default cache_path which
                is in the root of the user's directory.

        Returns:
            str: A string representation of the file path including the \
                filename.

        """
        if cache_path is None:
            cache_path = get_cache_path()

        fname = self.__class__.__name__ + uuid.uuid4().hex
        fpath = "{}.pkl".format(fname)
        path = os.path.join(cache_path, fpath)

        return path

    @staticmethod
    def _pickle_inline_repr(obj):
        """Generate a string representation of a pickle of an object.

        Args:
            obj: Any object

        Returns:
            str: The string representation of the pickle

        """
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL).hex()

    @staticmethod
    def _unpickle_inline_repr(pickle_str):
        """Goes from a string representation of a pickle to the object.

        Args:
            pickle_str (str): A pickle string

        Returns:
            object: A deserialized object

        """
        return pickle.loads(bytearray.fromhex(pickle_str))

    @classmethod
    def pickle_class_def(cls):
        """Pickle a class definition.

        Returns:
            dict: With a `pickle_class` key and the pickled class definition.

        """
        # Short circuit if statement if __module__ not in class
        is_internal = hasattr(cls, "__module__") and (
            "foreshadow" in getattr(cls, "__module__")
        )
        if not is_internal:
            return {"pickle_class": cls._pickle_inline_repr(cls)}
        else:
            return {}

    def dict_serialize(self, deep=True):
        """Serialize the init parameters (dictionary form) of a transformer.

        Args:
            deep (bool): If True, will return the parameters for this estimator
                recursively

        Returns:
            dict: The initialization parameters of the transformer.

        """
        return {"data": self.get_params(deep)}

    @classmethod
    def dict_deserialize(cls, data):
        """Deserialize the dictionary form of a transformer.

        Args:
            data: The dictionary to parse as the transformer is constructed.

        Returns:
            object: A re-constructed transformer

        """
        params = data["data"]
        pickle_class = data.get("pickle_class")
        if pickle_class is not None:
            pickle_class = cls._unpickle_inline_repr(pickle_class)
            return pickle_class(**params)
        else:
            # Cannot use set_params since steps is a required init arg
            # for Pipelines
            return cls(**params)

    def inline_serialize(self):
        """Convert transformer to hex pickle form inline in a dictionary form.

        Returns:
            A string representation of the pickle dump

        """
        return {"data": self._pickle_inline_repr(self)}

    @classmethod
    def inline_deserialize(cls, data):
        """Unpickle an inline pickled transformer.

        Args:
            data: The dictionary data of the transformer.

        Returns:
            object: The constructed transformer.

        """
        return cls._unpickle_inline_repr(data["data"])

    def disk_serialize(self, cache_path=None):
        """Convert transformer to pickle and save it disk in a cache directory.

        Args:
            cache_path (str): Override the default cache path which is in the
                root of the user directory

        Returns:
            str: The path the data was saved to.

        """
        fpath = self._pickle_cache_path(cache_path)
        with open(fpath, "wb+") as fopen:
            pickle.dump(self, fopen, protocol=pickle.HIGHEST_PROTOCOL)

        return {"data": fpath}

    @classmethod
    def disk_deserialize(cls, data):
        """Deserialize a transformer disk cache serialized.

        Args:
            data: The dictionary data of the transformer.

        Returns:
            object: The constructed transformer.

        """
        fpath = data["data"]
        with open(fpath, "rb") as fopen:
            return pickle.load(fopen)

    def serialize(self, method=None, name=None, **kwargs):
        """Serialize data as specified.

        If you would like to save the transformer parameters without saving
        its state in a human readable form, use `dict`. If you would like to
        save the transformer with its internal state use `inline` to
        save it in its hex form in the json. If you would like a more space
        efficient form save use `disk` to save it a cache directory in
        the root (~/.foreshadow/cache) that must be manually cleaned. Lastly,
        if the transformer being serialized is custom, then the class itself
        will be cached in pickle form and placed in the `pickle_class`
        attribute.

        Args:
            method (str): A choice between `json` and `pickle` to serialize a
                string.
            name (str): The name associated with the transformer. If not
                specified, a name will be derived if possible.
            **kwargs: The keyword arguments to pass to the serialization method

        Returns:
            str: The appropriate string representation of the serialization.

        """
        payload = super().serialize(method=method, **kwargs)
        instance_name = _retrieve_name(self) if name is None else name

        return {
            "name": instance_name if instance_name != "var" else None,
            **self.pickle_class_def(),
            **payload,
        }

    def to_json(self, path, **kwargs):
        """Save a serialized form a transformer to disk in json form.

        Args:
            path: The path to save the transformer
            **kwargs: Any further options to pass to serialize

        """
        with open(path, "w+") as fopen:
            json.dump(self.serialize(**kwargs), fopen, indent=2)

    @classmethod
    def from_json(cls, path):
        """Load a json representation of a transformer from disk.

        Args:
            path (str): The path to load the data from.

        Returns:
            The constructed object.

        """
        with open(path, "r") as fopen:
            return cls.deserialize(json.load(fopen))

    def to_yaml(self, path, **kwargs):
        """Save a serialized form of a transformer to disk in yaml form.

        Args:
            path: The path to save the transformer
            **kwargs: Any further options to pass to serialize

        """
        with open(path, "w+") as fopen:
            yaml.dump(self.serialize(**kwargs), fopen)

    @classmethod
    def from_yaml(cls, path):
        """Load a yaml representation of a transformer from disk.

        Args:
            path (str): The path to load the data from.

        Returns:
            The constructed object.

        """
        with open(path, "r") as fopen:
            return cls.deserialize(yaml.safe_load(fopen))


class PipelineSerializerMixin(ConcreteSerializerMixin):
    """An custom serialization method to allow pipelines serialization."""

    def dict_serialize(self, deep=False):
        """Serialize the init parameters (dictionary form) of a pipeline.

        Note:
            This recursively serializes the individual steps to facilitate a
            human readabel form.

        Args:
            deep (bool): If True, will return the parameters for this estimator
                recursively

        Returns:
            dict: The initialization parameters of the pipeline.

        """
        data = self.get_params(deep)
        steps = data.pop("steps")

        steps = []
        for name, transformer in self.steps:
            steps.append(
                transformer.serialize(method="dict", name=name, deep=False)
            )

        data["steps"] = steps

        return {"data": data}

    @classmethod
    def dict_deserialize(cls, data):
        """Deserialize the dictionary form of a pipeline.

        Args:
            data: The dictionary to parse as the pipeline is constructed.

        Returns:
            object: A re-constructed pipeline

        """
        ser_steps = data["data"]["steps"]

        deser_steps = []
        for step in ser_steps:
            deser_steps.append(
                (
                    step["name"],
                    _obj_deserializer_helper(step).deserialize(step),
                )
            )

        data["data"]["steps"] = deser_steps

        return super().dict_deserialize(data)


def _obj_deserializer_helper(data):
    """Handle the case when a custom object is pickled.

    Args:
        data: The dictionary form of the transformer.

    Returns:
        The constructed class.

    """
    pickle_class = data.get("pickle_class")
    if pickle_class is not None:
        transformer = ConcreteSerializerMixin._unpickle_inline_repr(
            pickle_class
        )
    else:
        transformer = get_transformer(data["class"])

    return transformer


def deserialize(data):
    """Allow the deserialization of any transformer.

    Args:
        data: The dictionary form of the transformer.

    Returns:
        object: The constructed transformer.

    """
    return _obj_deserializer_helper(data).deserialize(data)
