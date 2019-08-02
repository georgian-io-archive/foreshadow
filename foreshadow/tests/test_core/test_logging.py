from unittest import mock

import pytest


@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_get_logger_1(mock_open):
    """Test the correct functionality of get_logger.

    Args:
        n_gets: number of times to try getting a logger.

    """
    from foreshadow.logging import logging
    import logging as _logging

    logger = logging.get_logger()
    assert isinstance(logger, _logging._loggerClass)


@pytest.mark.parametrize("n_gets", [2, 3, 100])
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_get_logger_greater1(mock_open, n_gets):
    """Test the correct functionality of get_logger.

    Args:
        n_gets: number of times to try getting a logger.

    """
    from foreshadow.logging import logging

    logger = logging.get_logger()
    for _ in range(n_gets - 1):
        logger_ = logging.get_logger()
    assert logger == logger_


@pytest.mark.parametrize(
    "level", ["debug", "info", "warning", "error", "critical"]
)
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_get_log_fn(mock_open, caplog, level):
    """Test that log_fn logs properly to a stream.

    Args:
        caplog: captures logging output.
        level: the loggig level to test

    """
    from foreshadow.logging import logging

    logger = logging.get_logger()
    logger.setLevel(logging.LEVELS[level])
    test_print = "testing"
    log_fn = logging._get_log_fn(level)
    log_fn(test_print)
    assert caplog.record_tuples[0][2] == test_print


@pytest.mark.parametrize(
    "level,schema,outfile",
    [
        ("debug", "MetricSchema", "gui_data.txt"),
        ("info", "MetricSchema", "gui_data.txt"),
        ("warning", "MetricSchema", "gui_data.txt"),
        ("error", "MetricSchema", "gui_data.txt"),
        ("critical", "MetricSchema", "gui_data.txt"),
        ("debug", "MetricSchema", "random_file.txt"),
        ("debug", "MetricSchema", "gui_data.txt"),
    ],
)
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_log_and_gui(mock_open, caplog, level, schema, outfile):
    """Test log_and_gui logs and writes to gui.

    Args:
        mock_open: mock the open builtin call and test it being called
        caplog: captures logging output.
        level: the loggig level to test
        schema: schema to use for gui
        outfile: outfile to write to.

    """
    from foreshadow.logging import logging
    from foreshadow.utils.testing import dynamic_import

    logging.gui_fn.buffer_size = 1
    logging.gui_fn.first_write = True  # have to set as each parametrize
    # will use the previous call's gui_fn, meaning first_write will already
    # be set to False
    logging.gui_fn.outfile = outfile
    path = "foreshadow.logging.gui"
    schema = dynamic_import(schema, path)()
    schema_fields = schema.declared_fields
    details = {}
    for name, field in schema_fields.items():
        field_type = type(field).__name__.split(".")[-1]
        if field_type == "String":
            func = str
        elif field_type == "Float":
            func = float
        else:
            raise NotImplementedError(
                "field type: %s is not valid" % field_type
            )
        details[name] = func(1)
    logging.set_level("debug")
    msg = "test"
    logging.log_and_gui(level, msg, schema, details)
    assert caplog.record_tuples[0][2] == msg
    mock_open.assert_called_with(logging.gui_fn.outfile, "w")


@pytest.mark.parametrize(
    "level", ["debug", "info", "warning", "error", "critical"]
)
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_set_level(mock_open, level):
    """ Test ability to set logging level.

    Args:
        level: logging level to set.

    """
    from foreshadow.logging import logging

    logging.set_level(level)
    assert logging.get_logger().level == logging.LEVELS[level]


@pytest.mark.parametrize(
    "level", ["debug", "info", "warning", "error", "critical"]
)
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_simple(mock_open, caplog, level):
    """Test the usage of api methods to log for each level.

    Args:
        caplog: captures logging output.
        level: logging level to test.

    """
    from foreshadow.utils.testing import dynamic_import
    from foreshadow.logging import logging

    logging.set_level(level)
    log = dynamic_import(level, "foreshadow.logging.logging")
    msg = "test"
    log(msg)
    assert caplog.record_tuples[0][2] == msg


@pytest.mark.parametrize("name", ["name"])
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_wrap_func(mock_open, name):
    """Test wrapping of func usng wrap_func, including being called with name.

    Args:
        name: name for func.

    """
    from foreshadow.logging import logging as logging

    fn = mock.Mock()
    fn.return_value = "test"
    wrapped_func = logging._wrap_log(fn, name)
    assert wrapped_func.__name__ == name
    wrapped_func()
    fn.assert_called_with(name)


@pytest.mark.parametrize(
    "n_buffer_items,write_called_with", [(50, "w"), (0, "w"), (125, "a")]
)
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_sync_gui(mock_open, caplog, n_buffer_items, write_called_with):
    """Test that sync_gui performs final write operation correctly.

    Args:
        mock_open: mocks any open calls
        caplog: captures logging outputs
        n_buffer_items: number of items to write to buffer
        write_called_with: the file type of the write operation

    Returns:

    """
    from foreshadow.logging import logging as logging

    logging.gui_fn.first_write = True  # have to set as each parametrize
    # will use the previous call's gui_fn, meaning first_write will already
    # be set to False
    logging.gui_fn.buffer_size = 100
    for _ in range(n_buffer_items):
        logging.gui_fn.buffer.append("test")
        if len(logging.gui_fn.buffer) > logging.gui_fn.buffer_size:
            logging.gui_fn.write()
    logging.sync_gui()
    mock_open.assert_called_with("gui_data.txt", write_called_with)


@pytest.mark.parametrize(
    "n_buffer_items,buffer_val, expected_log",
    [
        (10, -1, "trying to set buffer size < 1"),
        (10, 5, "trying to set buffer < current buffer length"),
        (1000, 100, "trying to set buffer < current buffer length"),
    ],
)
@mock.patch("builtins.open", new_callable=mock.mock_open())
def test_buffer_setter(
    mock_open, caplog, n_buffer_items, buffer_val, expected_log
):
    """Test buffer setter.

    Args:
        mock_open: mocks any open calls
        caplog: captures logging outputs
        n_buffer_items: number of items to write to buffer
        buffer_val: the buffer size
        expected_log: expected logging statements

    """
    from foreshadow.logging import logging

    logging.set_level("debug")

    logging.gui_fn.first_write = True  # have to set as each parametrize
    # will use the previous call's gui_fn, meaning first_write will already
    # be set to False
    logging.gui_fn.buffer_size = 10000
    for _ in range(n_buffer_items):
        logging.gui_fn.buffer.append("test")
    logging.gui_fn.buffer_size = buffer_val
    mock_open.assert_called_with("gui_data.txt", "w")
    print(caplog.record_tuples)
    assert caplog.record_tuples[0][2].find(expected_log) != -1
