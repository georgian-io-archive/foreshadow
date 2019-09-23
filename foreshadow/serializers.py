"""Transformer serialization utilities."""

import json
import os
import pickle
import uuid

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import yaml

from foreshadow.utils import get_cache_path, get_transformer


jsonpickle_numpy.register_handlers()
_pickler = jsonpickle.pickler.Pickler()
_unpickler = jsonpickle.unpickler.Unpickler()


def _make_serializable(data, serialize_args={}):
    """Make sure that all arguments in a dictionary are "serializable".

    This means that all of they keys and values can be written to JSON or
    YAML form.

    Args:
        data (dict): A dictionary representation of a transformer.
        serialize_args (dict, optional): The arguments that should be passed to
            a serialize method if necessary.

    Returns:
        dict: A dictionary mirroring the input with any keys that need fixing
            updating.

    """
    try:
        json.dumps(data)
        return data
    except TypeError:
        if isinstance(data, dict):
            result = {
                k: _make_serializable(v, serialize_args=serialize_args)
                for k, v in data.items()
            }
        # elif hasattr(data, "__iter__"):  # I don't think __next__ is correct
        elif isinstance(data, (list, tuple)):
            result = [
                _make_serializable(v, serialize_args=serialize_args)
                for v in data
            ]
        else:
            # If the data argument is able to be serialized, then simply
            # serialize it using the same args that were passed into the top
            # level serialize method
            if hasattr(data, "serialize"):
                result = data.serialize(**serialize_args)
            else:
                result = _pickler.flatten(data)

        return result


def _make_deserializable(data):
    """Inverse of `_make_serializable` which reconstructs any complex objects.

    Args:
        data (dict): A dictionary with "flattened" or "serialized" elements


    Returns:
        dict: A dictionary with complex objects reconstructed as necessary.

    """
    if isinstance(data, dict):
        if any("py/" in s for s in data.keys()):
            return _unpickler.restore(data)
        if any("_method" == s for s in data.keys()):
            # TODO add test, watch out for keys like 'hash_method'
            return deserialize(data)
        else:
            new_data = {}
            for k, v in data.items():
                new_data[k] = _make_deserializable(v)
            return new_data
    elif isinstance(data, (list, tuple)):
        new_data = []
        for v in data:
            new_data.append(_make_deserializable(v))
        return new_data
    else:
        return data


def _pickle_cache_path(cls_name, cache_path=None):
    """Get the pickle cache path of a transformer.

    Uses a generated UUID and the class name to come up with a unique
    filename.

    Args:
        cls_name (str): Name of the class to be pickled
        cache_path (str, optional): override the default cache_path which
            is in the root of the user's directory.

    Returns:
        str: A string representation of the file path including the \
            filename.

    """
    if cache_path is None:
        cache_path = get_cache_path()

    fname = cls_name + uuid.uuid4().hex
    fpath = "{}.pkl".format(fname)
    path = os.path.join(cache_path, fpath)

    return path


def _pickle_inline_repr(obj):
    """Generate a string representation of a pickle of an object.

    Args:
        obj: Any object

    Returns:
        str: The string representation of the pickle

    """
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL).hex()


def _unpickle_inline_repr(pickle_str):
    """Goes from a string representation of a pickle to the object.

    Args:
        pickle_str (str): A pickle string

    Returns:
        object: A deserialized object

    """
    return pickle.loads(bytearray.fromhex(pickle_str))


def _obj_deserializer_helper(data):
    """Handle the case when a custom object is pickled.

    Args:
        data: The dictionary form of the transformer.

    Returns:
        The constructed class.

    """
    pickle_class = data.get("_pickled_class")
    if pickle_class is not None:
        transformer = _unpickle_inline_repr(pickle_class)
    else:
        transformer = get_transformer(data["_class"])

    return transformer


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
            if "_method" in kwargs:
                method = kwargs.pop("_method")
            else:
                method = self.DEFAULT_OPTION

        if method in self.OPTIONS:
            method_func = getattr(self, method + "_serialize")
            self.serialize_params = {"_method": method, **kwargs}
            payload = method_func(**kwargs)
        else:
            raise ValueError(
                "Serialization method must be one of {}".format(self.OPTIONS)
            )

        payload["_class"] = self.__class__.__name__
        payload["_method"] = method

        return payload

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
        method = data.pop("_method")  # TODO: add malformed serialization error
        _ = data.pop("_class", None)
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

    @classmethod
    def pickle_class_def(cls):
        """Pickle a class definition.

        Returns:
            dict: With a `pickle_class` key and the pickled class definition.

        """
        # Short circuit the below if statement, if __module__ is not in
        # the class
        is_internal = hasattr(cls, "__module__") and (
            "foreshadow" in getattr(cls, "__module__")
        )
        if not is_internal:
            return {"_pickled_class": _pickle_inline_repr(cls)}
        else:
            return {}

    def dict_serialize(self, deep=False):
        """Serialize the init parameters (dictionary form) of a transformer.

        Args:
            deep (bool): If True, will return the parameters for this estimator
                recursively

        Returns:
            dict: The initialization parameters of the transformer.

        """
        to_serialize = self.get_params(deep)
        return _make_serializable(
            to_serialize, serialize_args=self.serialize_params
        )

    @classmethod
    def dict_deserialize(cls, data):
        """Deserialize the dictionary form of a transformer.

        Args:
            data: The dictionary to parse as the transformer is constructed.

        Returns:
            object: A re-constructed transformer

        """
        params = _make_deserializable(data)
        pickle_class = params.pop("_pickled_class", None)
        if pickle_class is not None:
            pickle_class = _unpickle_inline_repr(pickle_class)
            return pickle_class(**params)
        else:
            # Cannot use set_params since steps is a required init arg
            # for Pipelines and therefore we cannot use default
            # init method (assuming no required args) to initialize
            # an instance then call set_params.
            if issubclass(cls, PipelineSerializerMixin):
                return cls(**params)
            else:
                ret_tf = cls()
                ret_tf.set_params(**params)

                return ret_tf

    def inline_serialize(self):
        """Convert transformer to hex pickle form inline in a dictionary form.

        Returns:
            A string representation of the pickle dump

        """
        return {"_pickled_obj": _pickle_inline_repr(self)}

    @classmethod
    def inline_deserialize(cls, data):
        """Unpickle an inline pickled transformer.

        Args:
            data: The dictionary data of the transformer.

        Returns:
            object: The constructed transformer.

        """
        return _unpickle_inline_repr(data["_pickled_obj"])

    def disk_serialize(self, cache_path=None):
        """Convert transformer to pickle and save it disk in a cache directory.

        Args:
            cache_path (str): Override the default cache path which is in the
                root of the user directory

        Returns:
            str: The path the data was saved to.

        """
        fpath = _pickle_cache_path(self.__class__.__name__, cache_path)
        with open(fpath, "wb+") as fopen:
            pickle.dump(self, fopen, protocol=pickle.HIGHEST_PROTOCOL)

        return {"_file_path": fpath}

    @classmethod
    def disk_deserialize(cls, data):
        """Deserialize a transformer disk cache serialized.

        Args:
            data: The dictionary data of the transformer.

        Returns:
            object: The constructed transformer.

        """
        fpath = data["_file_path"]
        with open(fpath, "rb") as fopen:
            return pickle.load(fopen)

    def serialize(self, method=None, **kwargs):
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
            **kwargs: The keyword arguments to pass to the serialization method

        Returns:
            str: The appropriate string representation of the serialization.

        """
        payload = super().serialize(method=method, **kwargs)

        return {**self.__class__.pickle_class_def(), **payload}

    def to_json(self, path, **kwargs):
        """Save a serialized form a transformer to disk in json form.

        Args:
            path: The path to save the transformer
            **kwargs: Any further options to pass to serialize

        """
        with open(path, "w+") as fopen:
            test = self.serialize(**kwargs)
            json.dump(test, fopen, indent=2)

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
            yaml.safe_dump(self.serialize(**kwargs), fopen)

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
        """Serialize a pipeline by serializing selected fields.

        Steps in the pipeline are reformatted as {"step_name": step}

        Args:
            deep: see super

        Returns:
            dict: serialized pipeline

        """
        to_serialize = {}
        all_params = self.get_params(deep=deep)
        to_serialize["memory"] = all_params.pop("memory", None)
        to_serialize["steps"] = all_params.pop("steps")
        serialized = _make_serializable(
            to_serialize, serialize_args=self.serialize_params
        )
        serialized["steps"] = [
            {step[0]: step[1]} for step in serialized["steps"]
        ]
        return serialized

    @classmethod
    def dict_deserialize(cls, data):
        """Deserialize pipeline from JSON.

        Steps in the pipeline are reformatted to [("step_name": step), ...]

        Args:
            data: the serailzied pipeline.

        Returns:
            a reconstructed pipeline

        """
        params = _make_deserializable(data)
        params["steps"] = [list(step.items())[0] for step in params["steps"]]
        return cls(**params)


def deserialize(data):
    """Allow the deserialization of any transformer.

    Args:
        data: The dictionary form of the transformer.

    Returns:
        object: The constructed transformer.

    """
    # We manually call this deserialize method so we can route to the correct
    # deserialize method (ie after knowing what the class is)
    return _obj_deserializer_helper(data).deserialize(data)
