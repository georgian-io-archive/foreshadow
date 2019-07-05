import pickle

from sklearn.externals import joblib


class SerializerMixin(object):
    """Mixin class for all transformers to be saved to disk"""

    def __get_json(self, **kwargs):
        pass

    def __get_pickle(self, **kwargs):
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    def serialize(self, method="json", **kwargs):
        """Serialize data to disk as necessary.

        Adds functionality to get_params.

        Args:
            method (str): A choice between `json`, `pickle`, and `joblib` to 
                serialize a string.
            **kwargs: The keyword arguments to pass to the 
                :meth:`sklearn.base.BaseEstimator.get_params` command.

        Returns:
            str: The appropriate string representaiton of the serialization.

        """
        options = {"json": self.__get_json, "pickle": self.__get_pickle}

        try:
            return options.get(method)(**kwargs)
        except KeyError:
            raise ValueError(
                "Serialization method must be one of {}".format(options.keys())
            )

    def deserialize(self):
        pass

    def to_disk(self):
        pass

    def from_disk(self):
        pass
