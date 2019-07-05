import re

import pandas as pd
import pytest

from foreshadow.column_sharer import ColumnSharer


simple_dataframe = pd.Series([i for i in range(10)])


@pytest.mark.parametrize("args,kwargs", [([], {})])
def test_column_sharer_create(args, kwargs):
    """Test arbitrary function can be converted to Metric using metric.

    Args:
        metric_fn: arbitrary metric function

    """
    from collections import MutableMapping

    cs = ColumnSharer(*args, **kwargs)
    assert isinstance(cs, MutableMapping)


@pytest.mark.parametrize(
    "key,expected_error1,expected_error2,expected_str1,expected_str2",
    [
        ("testing", None, KeyError, None, ".+already a registered key.+"),
        (
            "domain",
            KeyError,
            KeyError,
            ".+ already a predefined key.+",
            ".+ already a predefined key.+",
        ),
    ],
)
def test_column_sharer_register(
    key, expected_error1, expected_error2, expected_str1, expected_str2
):
    """Test registering key, ensuring key is on dict and error is raised.

    Runs the register_key function twice as some errors only occur on the
    second attempt.

    Args:
        key: key to register into ColumnSharer
        expected_error1: None if no error on first try, otherwise the expected
            error.
        expected_error2: None if no error on the second try, otherwise the
            expected error
        expected_str1: expected string for the error on first attempt
            expected
        expected_str2: expected string for the error on second attempt
            expected

    """
    cs = ColumnSharer()
    if expected_error1 is not None:
        with pytest.raises(expected_error1) as e1:
            cs.register_key(key)
    else:
        cs.register_key(key)
    if expected_error2 is not None:
        with pytest.raises(expected_error2) as e2:
            cs.register_key(key)
    else:
        cs.register_key(key)
    if expected_error1 is not None:
        if expected_str1 is not None:
            assert re.match(expected_str1, str(e1.value))
    if expected_error2 is not None:
        if expected_str2 is not None:
            assert re.match(expected_str2, str(e2.value))
    assert cs.store.get(key, None) is not None


@pytest.mark.parametrize(
    "key,expected",
    [
        ("test_key", ("test_key", None)),
        (["test_key", "test_column"], ("test_key", "test_column")),
        (["1", "2", "3"], KeyError),
    ],
)
def test_column_sharer_convert_key(key, expected):
    """Test that key conversion and error raising works as expected.

    Args:
        key: key to register into ColumnSharer
        expected: error if error, result if result.

    """
    cs = ColumnSharer()
    try:  # assume expected is an error
        if issubclass(expected, BaseException):  # this will fail if it isn't
            with pytest.raises(expected) as e:
                cs._convert_key(key)
            assert issubclass(e.type, expected)
    except TypeError:  # then expected will be the true result returned
        assert cs._convert_key(key) == expected


@pytest.mark.parametrize(
    "key,item_to_set,expected",
    [
        (["domain"], {}, {}),
        (["domain", "column"], {"column": 1}, 1),
        (["domain", "column"], {}, KeyError),  # can't get column info if
        # there is no column info
        (["domain", None], {}, {}),
    ],
)
def test_column_sharer_getitem(key, item_to_set, expected):
    """Test that getitem works for all valid key combinations or error raised.

    Args:
        key (list): key to access on ColumnSharer
        item_to_set: the item to set on the key as starting data. Dependent
            on the length of the key.
        expected: the expected result or error

    """
    cs = ColumnSharer()
    if len(key) == 1:
        cs.store[key[0]] = item_to_set
        try:  # assume expected is an error
            if issubclass(
                expected, BaseException
            ):  # this will fail if it isn't
                with pytest.raises(expected) as e:
                    cs[key[0]]
                assert issubclass(e.type, expected)
        except TypeError:  # then expected will be the true result returned
            assert cs[key[0]] == expected

    elif len(key) == 2:
        cs.store[key[0]] = item_to_set
        try:  # assume expected is an error
            if issubclass(
                expected, BaseException
            ):  # this will fail if it isn't
                with pytest.raises(expected) as e:
                    cs[key[0], key[1]]
                assert issubclass(e.type, expected)
        except TypeError:  # then expected will be the true result returned
            assert cs[key[0], key[1]] == expected

    else:
        raise NotImplementedError("test case not implemented")


@pytest.mark.parametrize(
    "key,key_to_register,expected",
    [
        ("domain", None, None),
        ("test", "test", None),
        ("test", None, KeyError)  # can't get column info if
        # there is no column info
    ],
)
def test_column_sharer_checkkey(key, key_to_register, expected):
    """Test that getitem works for all valid key combinations or error raised.

    Args:
        key (list): key to access on ColumnSharer
        key_to_register: key to register as starting info
        expected: the expected result or error

    """
    cs = ColumnSharer()
    if key_to_register is not None:
        cs.register_key(key_to_register)
    try:
        expected_is_exception = issubclass(expected, BaseException)  # this
        # will fail if expected is not an Exception
    except TypeError:
        expected_is_exception = False
    if expected_is_exception:
        with pytest.raises(expected) as e:
            cs.check_key(key)
        assert issubclass(e.type, expected)
    else:
        cs.check_key(key)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("domain", KeyError),
        (("domain", "haha"), KeyError),
        (("domain", "test"), None),
    ],
)
def test_column_sharer_delitem(key, expected):
    """Test that delitem works for all valid key combinations or error raised.

    Args:
        key (list): key to delete on ColumnSharer
        expected: the expected result or error

    """
    cs = ColumnSharer()
    cs.store["domain"] = {"test": True}
    if len(key) == 1 or isinstance(key, str):
        with pytest.raises(expected) as e:
            del cs[key]
        assert issubclass(e.type, expected)

    if len(key) == 2:
        if expected is not None:
            with pytest.raises(expected) as e:
                del cs[key[0], key[1]]
        else:
            del cs[key[0], key[1]]


@pytest.mark.parametrize(
    "store",
    [
        {"domain": {}, "intent": {}, "metastat": {}},
        {"domain": {"column1": [0, 1, 2]}},
        {
            "domain": {"column1": [0, 1, 2]},
            "intent": {"column1": [1, 2, 3], "column2": [1, 4, 6]},
            "metastat": {},
            "registered_key": {},
            "another_registered": {"column1": [1, 2, 3], "column2": True},
        },
    ],
)
def test_column_sharer_iter(store):
    """Test that iter iterates over entire internal dict properly.

    Args:
        store: the internal dictionary to use.

    """
    cs = ColumnSharer()
    for key in store:
        try:
            cs.register_key(key)
        except KeyError:
            pass
    cs.store = store
    expected = {}  # we try to recreate the internal dict using the keys
    for key in iter(cs):
        if expected.get(key[0], None) is None:
            expected[key[0]] = {}
        if key[1] is None:
            expected[key[0]] = cs[key]
        else:
            expected[key[0]][key[1]] = cs[key]
    assert expected == cs.store


@pytest.mark.parametrize(
    "key,item_to_set,expected",
    [
        (["domain"], {}, {}),
        (["domain", "column"], [1, 2, 3], [1, 2, 3]),
        (["random_column", "column"], None, KeyError),  # can't get column
        # info if there is no column info
        (["domain", None], {}, {}),
        (["random_column"], None, KeyError),  # can't get column
        # info if there is no column info
    ],
)
def test_column_sharer_setitem(key, item_to_set, expected):
    """Test that getitem works for all valid key combinations or error raised.

    Args:
        key (list): key to access on ColumnSharer
        item_to_set: the item to set on the key as starting data. Dependent
            on the length of the key.
        expected: the expected result or error

    """
    cs = ColumnSharer()
    if len(key) == 1:
        try:  # assume expected is an error
            if issubclass(
                expected, BaseException
            ):  # this will fail if it isn't
                with pytest.raises(expected) as e:
                    cs[key[0]] = item_to_set
                assert issubclass(e.type, expected)
        except TypeError:  # then expected will be the true result returned
            cs[key[0]] = item_to_set
            assert cs[key[0]] == expected

    elif len(key) == 2:
        try:  # assume expected is an error
            if issubclass(
                expected, BaseException
            ):  # this will fail if it isn't
                with pytest.raises(expected) as e:
                    cs[key[0], key[1]] = item_to_set
                assert issubclass(e.type, expected)
        except TypeError:  # then expected will be the true result returned
            cs[key[0], key[1]] = item_to_set
            print(cs.store)
            assert cs[key[0], key[1]] == expected

    else:
        raise NotImplementedError("test case not implemented")


@pytest.mark.parametrize(
    "store,expected",
    [
        ({"domain": {}, "intent": {}, "metastat": {}}, 0),
        ({"domain": {"column1": [0, 1, 2]}}, 1),
        (
            {
                "domain": {"column1": [0, 1, 2]},
                "intent": {"column1": [1, 2, 3], "column2": [1, 4, 6]},
                "metastat": {},
                "registered_key": {},
                "another_registered": {"column1": [1, 2, 3], "column2": True},
            },
            3,
        ),
    ],
)
def test_column_sharer_len(store, expected):
    """Test that iter iterates over entire internal dict properly.

    Args:
        store: the internal dictionary to use.

    """
    cs = ColumnSharer()
    cs.store = store
    for key in store:
        try:
            cs.register_key(key)
        except KeyError:
            pass
    assert len(cs) == expected
