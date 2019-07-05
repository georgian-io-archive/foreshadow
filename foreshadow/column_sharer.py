"""Cache utility for foreshadow pipeline workflow to share data."""
from collections import MutableMapping, defaultdict


class ColumnSharer(MutableMapping):
    """main cache-class to be used as single-instance to share data."""

    def __init__(self, *args, **kwargs):
        super(ColumnSharer, self).__init__(*args, **kwargs)
        self.store = defaultdict(lambda: {})  # will have a nested dict for
        # every key, which holds {column: key-column info}
        acceptable_keys = {"intent": True, "domain": True, "metastat": True}
        for key in acceptable_keys:
            self.store[key] = {}
        self.__acceptable_keys = defaultdict(lambda: False, acceptable_keys)
        self.__registered_keys = defaultdict(lambda: False)

    def __getitem__(self, key_list):
        """Override getitem to support multi key accessing simultaneously.

        Args:
            key_list: list of keys passed

        Returns:
            [key] or [key][column] from internal dict.

        Raises:
              e: exception raised by dict on invalid access. Likely
                  a KeyError.

        """
        # first, get the item from the dict
        key, column = self._convert_key(key_list)
        self.check_key(key)
        key_dict = self.store[key]
        if column is not None:  # then get the column if requested
            try:
                return key_dict[column]
            except Exception as e:
                raise e
        return key_dict  # otherwise return all the columns

    def __setitem__(self, key_list, value):
        """Enable multi key setting simultaneously.

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

        Raises:
            KeyError: if not a valid key (predefined or registered)

        """
        if not self.__acceptable_keys[key] and not self.__registered_keys[key]:
            raise KeyError(
                "key {} is not a valid key. Please register it "
                "first".format(key)
            )
        if self.__registered_keys[key]:
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

        Returns:
            Number of columns, duplicates counted.

        """
        length = 0
        for key in self.store:
            length += len(self.store[key])
        return length

    def register_key(self, key):
        """Register a non pre-defined key as acceptable.

        Must be called before using a key unless it is in the list of
        predefined acceptable keys (self.__acceptable_keys).

        Args:
            key: key to register

        Raises:
            KeyError: if key already set as predefined or registered

        """
        if self.__acceptable_keys[key]:
            raise KeyError("key: '{}' is already a predefined key".format(key))
        if self.__registered_keys[key]:
            raise KeyError("key: '{}' is already a registered key".format(key))
        self.store[key] = {}
        self.__registered_keys[key] = True
