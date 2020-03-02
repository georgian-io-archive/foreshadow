import pandas as pd
import pytest

from foreshadow.intents import IntentType
from foreshadow.utils import AcceptedKey, Override


simple_dataframe = pd.Series([i for i in range(10)])


@pytest.mark.parametrize("args,kwargs", [([], {})])
def test_cache_manager_create(args, kwargs):
    """Test creation of a ColumnSharer object.

    Args:
        args: args to ColumnSharer init
        kwargs: kwargs to ColumnSharer init

    """
    from collections import MutableMapping
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager(*args, **kwargs)
    assert isinstance(cs, MutableMapping)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("test_key", ("test_key", None)),
        (["test_key", "test_column"], ("test_key", "test_column")),
        (["1", "2", "3"], KeyError),
    ],
)
def test_cache_manager_convert_key(key, expected):
    """Test that key conversion and error raising works as expected.

    Args:
        key: key to register into ColumnSharer
        expected: error if error, result if result.

    """
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
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
def test_cache_manager_getitem(key, item_to_set, expected):
    """Test that getitem works for all valid key combinations or error raised.

    Args:
        key (list): key to access on ColumnSharer
        item_to_set: the item to set on the key as starting data. Dependent
            on the length of the key.
        expected: the expected result or error

    """
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
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
    "key,expected",
    [
        ("domain", None),
        ("test", "WARNING")  # can't get column info if
        # there is no column info
    ],
)
def test_cache_manager_checkkey(capsys, key, expected):
    """Test that getitem works for all valid key combinations.

    Args:
        capsys: captures stdout and stderr. Pytest fixture.
        key (list): key to access on ColumnSharer
        expected: the expected result or error

    """
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
    cs.check_key(key)
    out, err = capsys.readouterr()
    if expected is not None:
        assert out.find(expected) != -1
    else:
        assert len(out) == 0  # nothing in out.


@pytest.mark.parametrize(
    "key,expected",
    [
        ("domain", KeyError),
        (("domain", "haha"), KeyError),
        (("domain", "test"), None),
    ],
)
def test_cache_manager_delitem(key, expected):
    """Test that delitem works for all valid key combinations or error raised.

    Args:
        key (list): key to delete on ColumnSharer
        expected: the expected result or error

    """
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
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
def test_cache_manager_iter(store):
    """Test that iter iterates over entire internal dict properly.

    Args:
        store: the internal dictionary to use.

    """
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
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
    "key,item_to_set,expected,warning",
    [
        (["domain"], {}, {}, False),
        (["domain", "column"], [1, 2, 3], [1, 2, 3], False),
        (["random_column", "column"], None, None, True),  # return None and
        # print warning
        (["domain", None], {}, {}, False),
        (["random_column"], None, None, True),  # return None and
        # print warning
    ],
)
def test_cache_manager_setitem(capsys, key, item_to_set, expected, warning):
    """Test that getitem works for all valid key combinations or error raised.

    Args:
        capsys: captures stdout and stderr. Pytest fixture.
        key (list): key to access on ColumnSharer
        item_to_set: the item to set on the key as starting data. Dependent
            on the length of the key.
        expected: the expected result or error
        warning: True to check if should raise warning. False to not.

    """
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
    if len(key) == 1:
        cs[key[0]] = item_to_set
        assert cs[key[0]] == expected
        if warning:
            out, err = capsys.readouterr()
            assert out.find("WARNING") != -1

    elif len(key) == 2:
        cs[key[0], key[1]] = item_to_set
        print(cs.store)
        assert cs[key[0], key[1]] == expected
        if warning:
            out, err = capsys.readouterr()
            assert out.find("WARNING") != -1

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
            5,
        ),
    ],
)
def test_cache_manager_len(store, expected):
    """Test that iter iterates over entire internal dict properly.

    Args:
        store: the internal dictionary to use.

    """
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
    cs.store = store
    assert len(cs) == expected


def test_cache_manager_has_or_has_no_override():
    """Test that iter iterates over entire internal dict properly.

    Args:
        store: the internal dictionary to use.

    """
    from foreshadow.cachemanager import CacheManager

    cs = CacheManager()
    cs[AcceptedKey.OVERRIDE][
        "_".join([Override.INTENT, "col1"])
    ] = IntentType.CATEGORICAL
    assert cs.has_override()

    cs2 = CacheManager()
    assert not cs2.has_override()
