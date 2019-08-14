"""Cache utility for foreshadow pipeline workflow to share data."""
import pprint
from collections import MutableMapping, defaultdict

from foreshadow.serializers import ConcreteSerializerMixin


# TODO: Make this multi processor safe using managers


class PrettyDefaultDict(defaultdict):
    """A default dict wrapper that allows simple printing."""

    __repr__ = dict.__repr__


class ColumnSharer(MutableMapping, ConcreteSerializerMixin):
    """Main cache-class to be used as single-instance to share data.

    Note:
        This object is not thread safe for reads but is thread safe for writes.

    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: __iter__
    .. automethod:: __len__

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store = PrettyDefaultDict(lambda: PrettyDefaultDict(lambda: None))
        # will have a nested PrettyDefaultDict for every key, which holds
        # {column: key-column info} and gives None by default. It is the users
        # responsibility to make sure returned values are useful.
        acceptable_keys = {
            "intent": True,
            "domain": True,
            "metastat": True,
            "graph": True,
        }
        self.__acceptable_keys = PrettyDefaultDict(
            lambda: False, acceptable_keys
        )

    def get_params(self, deep=False):
        """Get parameters for the column sharer, which is the store in this case.

        Args:
            deep: Not used for this class. Default to False.

        Returns:
            Params for the column_sharer.

        """
        # Since the users are not supposed to change the acceptable_keys, I'm
        # inclined to not show them but we do need to show all of them in the
        # documentation.
        return {"store": self.store}

    def set_params(self, **params):
        """Set the parameters of the column sharer.

        Args:
            **params: params to set

        Returns:
            The column sharer with stores set.

        """
        for key in params["store"]:
            self[key] = params["store"][key]

        return self

    def __getitem__(self, key_list):
        """Override getitem to support multi key accessing simultaneously.

        You can access this data structure in two ways. The first is with
        the knowledge that the internal implementation uses nested dicts:
        ::

        >>> cs = ColumnSharer()
        >>> cs['domain']['not_a_column']
        None

        which will give you None by default with no previous value.
        You may also multi-access, treating it as a matrix:

        >>> cs = ColumnSharer()
        >>> cs['domain', 'not_a_column']
        None

        See __setitem__ for more details.

        Args:
            key_list: list of keys passed

        Returns:
            [key] or [key][column] from internal dict.

        """
        # first, get the item from the dict
        key, column = self._convert_key(key_list)
        self.check_key(key)
        key_dict = self.store[key]
        if column is not None:  # then get the column if requested
            return key_dict[column]
        return key_dict  # otherwise return all the columns

    def __setitem__(self, key_list, value):
        """Enable multi key setting simultaneously.

        You can set items on this data structure in two ways. The first is
        with  the knowledge that the internal implementation uses nested dicts
        ::

        >>> cs = ColumnSharer()
        >>> cs['domain']['not_a_column'] = 1
        >>> cs['domain']['not_a_column']
        1

        which will give you None by default with no previous value.
        You may also multi-access, treating it as a matrix:

        >>> cs = ColumnSharer()
        >>> cs['domain', 'not_a_column'] = 1
        >>> cs['domain']['not_a_column']
        1

        See __getitem__ for more details.

        Args:
            key_list: list of keys passed
            value: value to set, dict if just the key and value if key, column

        """
        key, column = self._convert_key(key_list)
        self.check_key(key)
        if column is None:  # setting the value for the entire key
            self.store[key] = value
        else:  # setting a particular column's value for a given key.
            self.store[key][column] = value

    def check_key(self, key):
        """Check they passed key to see if it is a valid key.

        Args:
            key: the key passed to this object.

        """
        if not self.__acceptable_keys[key]:
            print(
                "WARNING: the key {} is not an accepted key and relying on "
                "information here to exist at runtime could be "
                "dangerous".format(key)
            )  # TODO replace with logging

    @staticmethod
    def _convert_key(key):
        """Convert input key to internal structure; internal method.

        The supported key accessing methods are:
            (str), where it is the type of data
            (str, str), where it is (type of data, column)


        Args:
            key: the key passed to this object

        Returns:
            key, column based on the input format

        Raises:
            KeyError: if incorrect input format

        """
        if isinstance(key, str):  # technically doesn't need to be a string,
            # just not an array type.
            return key, None
        elif isinstance(key, (list, tuple)):
            if len(key) == 2:
                return key[0], key[1]
        raise KeyError(
            "input format of: {} is not a supported " "format.".format(key)
        )

    def __delitem__(self, key_list):
        """Enable deletion by column or by key.

        Args:
            key_list: key, or key and column

        Raises:
            KeyError: Trying to delete an entire key of data at once.

        """
        key, column = self._convert_key(key_list)
        self.check_key(key)
        if column is None:
            raise KeyError(
                "You cannot delete an entire key of data. Please "
                "pass a key and a column."
            )
        del self.store[key][column]

    def __iter__(self):
        """Will return list of (key, column) tuples ordered by key.

        Column will be None if none exist, in line with the __getitem__
        usage which will ignore it.

        Returns:
            iterator for internal nested dict structure. Will be ordered by
            key and then column.

        """
        # handle nested
        iterator = []
        for key in self.store:
            key_iter = self.store[key].keys()
            if len(key_iter) == 0:
                iterator.extend([(key, None)])
            else:
                iterator.extend([(key, x) for x in key_iter])
        return iter(iterator)

    def __len__(self):
        """Length of internal dict as the number of columns across keys.

        None unique. Aka, with 3 keys each with 5 columns, will return 15,
        even if the columns are the same.

        Returns:
            Number of columns, duplicates counted.

        """
        return sum([len(self.store[key]) for key in self.store])

    def __str__(self):
        """Get a string representation of the internal store.

        Returns:
            A pretty printed version of the internal store.

        """
        return pprint.pformat(self.store, indent=2)
