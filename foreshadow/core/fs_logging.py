"""Logging methods interfacing with a potential GUI system."""
import atexit as _atexit
import logging as _logging
import sys as _sys
import threading as _threading


_logger = None
_logger_lock = _threading.Lock()

_this_module = _sys.modules[__name__]

LEVELS = {
    "debug": _logging.DEBUG,
    "info": _logging.INFO,
    "warning": _logging.WARNING,
    "error": _logging.ERROR,
    "critical": _logging.CRITICAL,
}


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
        _my_logger = _logging.getLogger("foreshadow")

        _interactive = False
        try:
            # Only defined in interactive shells.
            _interactive = True if _sys.ps1 else _interactive
        except AttributeError:
            # check python -i
            _interactive = _sys.flags.interactive

        if _interactive:
            _my_logger.setLevel(LEVELS["info"])
        else:
            _my_logger.setLevel(LEVELS["warning"])
        _stream_target = _sys.stderr

        # Add Stream Handler based on if interactive or not.
        _handler = _logging.StreamHandler(_stream_target)
        _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
        _my_logger.addHandler(_handler)

        _logger = _my_logger
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
    """Single-instance object to handle buffered writing to file."""

    # TODO turn this into synchronous Queue based writing.
    def __init__(
        self, buffer_size=100, outfile="gui_data.txt", overwrite=True
    ):
        super(SyncWrite, self).__init__()
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
        # perform write operation to some nosql database or file
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


def _wrap_log(func, level):
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


@_atexit.register
def sync_gui():
    """Write anything left in gui buffer to file.

    This is called as python is exiting to ensure that any last calls to
    gui_fn that don't result in buffer_size being reached still write to the
    file. As well, joins for threads can be called here.
    """
    gui_fn.write()
    log_fn = _get_log_fn("debug")
    log_fn("asynchronous gui thread finished.")
