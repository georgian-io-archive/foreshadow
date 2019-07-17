"""Transformer serialization utilities."""

import inspect
import json
import os
import pickle
import uuid

import yaml

from foreshadow.exceptions import TransformerNotFound
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

class BaseTransformerSerializer():

    OPTIONS = []
    DEFAULT_OPTION = None
    DEFAULT_KWARGS = {}

    def serialize(self, method=None, **kwargs):
        if method is None:
            method = self.DEFAULT_OPTION

        default_kwargs = self.DEFAULT_KWARGS.get(method)
        if default_kwargs is not None:
            # Filter kwarg overrides and apply them to the kwargs before
            # before calling the serialization method
            kwargs = {
                **{
                    k: v
                    for k,v in default_kwargs.items()
                    if k not in kwargs
                },
                **kwargs
            }

        if method in self.OPTIONS:
            method_func = getattr(self, method+'_serialize')

            payload = method_func(**kwargs)
        else:
            raise ValueError(
                "Serialization method must be one of {}".format(self.OPTIONS)
            )

        return {
            'class': self.__class__.__name__,
            'method': method,
            **payload
        }

    @classmethod
    def deserialize(cls, data):
        method = data['method'] # TODO add malformed serialization error
        if  method in cls.OPTIONS:
            method_func = getattr(cls, method+'_deserialize')
            return method_func(data)
        else:
            raise ValueError(
                "Deserialization method must be one of {}".format(options.keys())
            )

class ConcreteSerializerMixin(BaseTransformerSerializer):
    """Mixin class that provides convenience serialization methods."""

    OPTIONS = [
        'dict',
        'inline',
        'disk'
    ]
    DEFAULT_OPTION = 'dict'

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
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL).hex()

    @staticmethod
    def _unpickle_inline_repr(pickle_str):
        return pickle.loads(bytearray.fromhex(pickle_str))

    @classmethod
    def pickle_class_def(cls):
        # Short circuit if statement if __module__ not in class
        is_internal = (
            hasattr(cls, '__module__')
            and ('foreshadow' in getattr(cls, '__module__'))
        )
        if not is_internal:
            return {
                'pickle_class': cls._pickle_inline_repr(cls)
            }
        else:
            return {}

    @classmethod
    def unpickle_class_def(cls):
        pass

    def dict_serialize(self, deep=True):
        """Get the params of a transformer, its initialization state.

        Args:
            deep (bool): If True, will return the parameters for this estimator
                recursively

        Returns:
            dict: The initialization parameters of the transformer.

        """

        return {
            'data': self.get_params(deep)
        }

    @classmethod
    def dict_deserialize(cls, data):
        params = data['data']
        pickle_class = data.get('pickle_class')
        if pickle_class is not None:
            pickle_class = cls._unpickle_inline_repr(pickle_class)
            return pickle_class(**params)
        else:
            # Cannot use set_params since steps is a required init arg
            # for Pipelines
            return cls(**params)

    def inline_serialize(self):
        """Convert transformer to pickle then to a hex form.
    
        Args:
            protocol: A pickle compression number

        Returns:
            A string representation of the pickle dump

        """
        return {
            'data': self._pickle_inline_repr(self)
        }

    @classmethod
    def inline_deserialize(cls, data):
        return cls._unpickle_inline_repr(data['data'])

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

        return {
            'data': fpath
        }

    @classmethod
    def disk_deserialize(cls, data):
        fpath = data["data"]
        with open(fpath, "rb") as fopen:
            return pickle.load(fopen)

    def serialize(self, method=None, name=None, **kwargs):
        """Serialize data as specified.

        If you would like to save the transformer parameters without saving
        its state in a human readable form, use `json`. If you would like to
        save the transformer with its internal state use `pickle_inline` to
        save it in its hex form in the json. If you would like a more space
        efficient form save use `pickle_disk` to save it a cache directory in
        the root (~/.foreshadow/cache) that must be manually cleaned. Lastly,
        if the transformer being serialized is custom, then the class itself
        will be cached in pickle form and placed in the `transformer`
        attribute. `custom_` will be pre-pended to the method name. Note, this
        is only applicable when the `method` is `json`.

        Note:
            See the individual serialization methods
            `get_json <foreshadow.core.SerializerMixin.get_json>` and
            `get_pickle <foreshadow.core.SerializerMixin.get_pickle>` for
            additional keyword arguments that are passed through the serialize
            method to those respective methods.

        Args:
            method (str): A choice between `json` and `pickle` to serialize a
                string.
            **kwargs: The keyword arguments to pass to the serialization method

        Returns:
            str: The appropriate string representation of the serialization.

        Raises:
            ValueError: If the serialization is not of the allowable options.

        """

        payload = super().serialize(method=method, **kwargs)

        instance_name = _retrieve_name(self) if name is None else name

        print(f"Here I am: {method} ")

        return {
            "name": instance_name if instance_name != "var" else None,
            **self.pickle_class_def(),
            **payload
        }

    def to_json(self, path, **kwargs):
        """Save a serialized form a transformer to disk.

        Args:
            path: The path to save the transformer
            **kwargs: Any further options to pass to serialize

        """
        with open(path, "w+") as fopen:
            json.dump(self.serialize(**kwargs), fopen, indent=2)

    @classmethod
    def from_json(cls, path):
        """Load A json representation of a transformer from disk.
 
        Args:
            path (str): The path to load the data from.
 
        Returns:
            The constructed object.
 
        """
        with open(path, "r") as fopen:
            return cls.deserialize(json.load(fopen))


    def to_yaml(self, path, **kwargs):
        """Save a serialized form a transformer to disk.

        Args:
            path: The path to save the transformer
            **kwargs: Any further options to pass to serialize

        """
        with open(path, "w+") as fopen:
            yaml.dump(self.serialize(**kwargs), fopen)

    @classmethod
    def from_yaml(cls, path):
        """Load A json representation of a transformer from disk.
    
        Args:
            path (str): The path to load the data from.
    
        Returns:
            The constructed object.
    
        """
        with open(path, "r") as fopen:
            return cls.deserialize(yaml.safe_load(fopen))

class PipelineSerializerMixin(ConcreteSerializerMixin):

    DEFAULT_KWARGS = {
        'dict': {
            'deep': False
        }
    }

    def dict_serialize(self, deep=True):
        data = self.get_params(deep)
        steps = data.pop('steps')

        steps = []
        for name, transformer in self.steps:
            steps.append(transformer.serialize(method='dict', name=name, deep=False))

        data['steps'] = steps

        return {
            'data': data
        }

    @classmethod
    def dict_deserialize(cls, data):
        ser_steps = data['data']['steps']

        deser_steps = []
        for step in ser_steps:
            deser_steps.append(
                (
                    step['name'],
                    _obj_deserializer_helper(step).deserialize(step)
                )
            )

        data['data']['steps'] = deser_steps

        return super().dict_deserialize(data)


def _obj_deserializer_helper(data):
    pickle_class = data.get('pickle_class')
    if pickle_class is not None:
        transformer = ConcreteSerializerMixin._unpickle_inline_repr(pickle_class)
    else:
        transformer = get_transformer(data['class'])

    return transformer

def deserialize(data):
    return _obj_deserializer_helper(data).deserialize(data)

if __name__ == "__main__":
    import numpy as np
    from foreshadow.transformers.core import SerializablePipeline
    from foreshadow.transformers.concrete import StandardScaler

    from sklearn.preprocessing import StandardScaler as sk_ss

    class SKSS(sk_ss, ConcreteSerializerMixin):
        pass

    test = np.arange(10).reshape((-1, 1))


#     skss = SKSS()
#     skss.fit(test)
#     ser = skss.serialize(method='disk')
#     deser = deserialize(ser)
# 
#     skss.to_json('test.yml')
#     deser2 = SKSS.from_json('test.yml')

    p = SerializablePipeline([('ss', StandardScaler())])
 
    p.fit(test)
    ser = p.serialize(method='disk')
    deser = deserialize(ser)

#    ss = StandardScaler()
#
#    ss.fit(test)
#    ser = ss.serialize()
#    deser = deserialize(ser)

    import pdb; pdb.set_trace()
