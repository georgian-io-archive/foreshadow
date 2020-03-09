"""Logging methods interfacing with a potential GUI system.

This module exposes logging.debug, logging.info, logging.warning,
logging.error, logging.critical as methods of this module. They can be
called as follows:
import logging
logging.debug(msg, *args, **kwargs)
and exactly mimic that of python's logging module.
"""
import atexit
import logging
import sys
import threading


_logger = None
_logger_lock = threading.Lock()

_this_module = sys.modules[__name__]

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}  # :py:attr:`foreshadow.core.logging.levels`

HIGHEST_LEVEL = "critical"
LOWEST_LEVEL = "debug"
LOGGING_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(" "levelname)s - %(process)d - %(" "message)s"
)


def get_logger():
    """Return Foreshadow logger instance.

    Will create and setup if needed, else will return the previously setup
    logger.

    Returns:
        foreshadow logger instance.

    """
    global _logger

    # if already created, don't acquire lock
    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger is not None:
            return _logger

        # Get scoped Foreshadow logger.
        my_logger = logging.getLogger("foreshadow")

        # interactive = False
        # if hasattr(sys, "ps1"):
        #     interactive = True
        #     # check python -i
        # elif hasattr(sys.flags, "interactive"):
        #     interactive = sys.flags.interactive

        # if interactive:
        #     my_logger.setLevel(LEVELS["info"])
        # else:
        #     my_logger.setLevel(LEVELS["warning"])
        my_logger.setLevel(LEVELS["info"])
        stream_target = sys.stderr

        # Add Stream Handler based on if interactive or not.
        handler = logging.StreamHandler(stream_target)
        # handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
        handler.setFormatter(LOGGING_FORMATTER)
        my_logger.addHandler(handler)

        _logger = my_logger
        return _logger

    finally:
        _logger_lock.release()


def _get_log_fn(level):
    """Get the logging function for a particular method from this module.

    Used internally in this script as the module will not yet have the
    methods set.

    Args:
        level: The corresponding log function level. E.g. 'debug'

    Returns:
        The log function callable.

    Raises:
        ValueError: if logging level does not have a logging function.

    """
    log_fn = getattr(_this_module, level, None)
    if log_fn is None:
        raise ValueError("level: '{}' is not a valid logging level.")
    return log_fn


class SyncWrite(object):
    """Single-instance object to handle buffered writing to file.

    Dummy class until we get to GUI.
    """

    # TODO turn this into synchronous Queue based writing.
    def __init__(
        self, buffer_size=100, outfile="gui_data.txt", overwrite=True
    ):
        self.threads = []
        self._buffer_size = buffer_size
        self.outfile = outfile
        self.first_write = overwrite
        self.buffer = []
        self.writing = False

    @property
    def buffer_size(self):
        """How long to wait before writing.

        Returns:
            buffer size as integer

        """
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        """Set buffer size to value and ensure it makes sense.

        Args:
            value: integer to set buffer_size to.

        """
        if value < 1:
            log_fn = _get_log_fn("warning")
            log_fn("trying to set buffer size < 1. Automatically setting to 1")
            value = 1
        if value < len(self.buffer):
            log_fn = _get_log_fn("warning")
            log_fn(
                "trying to set buffer < current buffer length. "
                "Automatically writing buffer to file before setting."
            )
            self.write()
        self._buffer_size = value

    def write(self):
        """Write buffer (gui calls) to file."""
        log_fn = _get_log_fn("debug")
        log_fn("writing gui buffer with: %d msgs to file" % len(self.buffer))
        io_type = "w" if self.first_write else "a"
        with open(self.outfile, io_type) as outfile:
            for msg in self.buffer:
                outfile.write(msg)
                outfile.write("\n")
        self.first_write = False
        self.buffer = []

    def __call__(self, schema, details):
        """Write to Gui. Will store in buffer until it reaches buffer_size.

        Args:
              schema: schema from foreshadow.core.gui to serialize
              details: dict of values corresponding to schema.

        """
        msg = str(schema.load(details).data)
        # TODO perform write operation to some nosql database or file
        if len(self.buffer) >= self.buffer_size - 1:  # -1 as append.
            self.buffer.append(msg)
            self.write()
        else:
            self.buffer.append(msg)


gui_fn = SyncWrite()


def set_level(level_name):
    """Set logger level.

    Args:
        level_name: level to set logger to. Must be in LEVELS.

    """
    level = LEVELS[level_name]
    get_logger().setLevel(level)


def _log(level, msg, *args, **kwargs):
    """Get log function of this module and log msg to it.

    Args:
        level: level of logging function
        msg: msg to write
        *args: args to function call
        **kwargs: kwargs to function call

    Raises:
        NotImplementedError: if logging function is not implemented for level.

    """
    log_fn = getattr(get_logger(), level, None)
    if log_fn is None:
        raise NotImplementedError(
            "please implement the logging function "
            "for level: '{}'".format(level)
        )
    log_fn(msg, *args, **kwargs)


def _wrap_log(func, level):  # noqa: D202
    """Wrap func to be called with level parameter and appear as that level.

    func should be _log.

    Args:
        func: function to wrap
        level: logging level to use

    Returns:
        wrapped function that appears to have name specified by level.

    """

    def wrapped_func(*args, **kwargs):
        return func(level, *args, **kwargs)

    wrapped_func.__name__ = level
    return wrapped_func


for level in LEVELS:  # dynamicaally expose the logging methods for each level
    # as functions of this module.
    log = _wrap_log(_log, level)
    setattr(_this_module, level, log)


def debug(*args, **kwargs):
    """Log debug message.

    Manually overriding so that this method is explicitly a part of this
    module.

    Args:
        *args: To logging.debug
        **kwargs: To logging.debug

    Returns:
        logging.debug

    """
    log = _wrap_log(_log, "debug")
    return log(*args, **kwargs)


def info(*args, **kwargs):
    """Log info message.

    Manually overriding so that this method is explicitly a part of this
    module.

    Args:
        *args: To logging.info
        **kwargs: To logging.info

    Returns:
        logging.info

    """
    log = _wrap_log(_log, "info")
    return log(*args, **kwargs)


def warning(*args, **kwargs):
    """Log warning message.

    Manually overriding so that this method is explicitly a part of this
    module.

    Args:
        *args: To logging.info
        **kwargs: To logging.info

    Returns:
        logging.info

    """
    log = _wrap_log(_log, "warning")
    return log(*args, **kwargs)


def error(*args, **kwargs):
    """Log error message.

    Manually overriding so that this method is explicitly a part of this
    module.

    Args:
        *args: To logging.info
        **kwargs: To logging.info

    Returns:
        logging.info

    """
    log = _wrap_log(_log, "error")
    return log(*args, **kwargs)


def log_and_gui(level, msg, gui_details, gui_schema, *args, **kwargs):
    """Log msg to gui at specific level and write gui_details under gui_schema.

    Level and msg correspond to standard logging level and msg. Gui_details
    is a dict corresponding to the specified gui_schema (in
    foreshadow.core.gui) that will be serialized to and written for the gui
    to interpret.

    Args:
        level: logging level.
        msg: message to write.
        gui_details: the dict of field, value pairs
        gui_schema: the marshmallow schemato use
        *args: args to logger
        **kwargs: kwargs to logger

    """
    _log(level, msg, *args, **kwargs)
    gui_fn(gui_details, gui_schema)


@atexit.register
def sync_gui():
    """Write anything left in gui buffer to file.

    This is called as python is exiting to ensure that any last calls to
    gui_fn that don't result in buffer_size being reached still write to the
    file. As well, joins for threads can be called here.
    """
    if len(gui_fn.buffer) > 0:
        gui_fn.write()
    log_fn = _get_log_fn("debug")
    log_fn("asynchronous gui thread finished.")
