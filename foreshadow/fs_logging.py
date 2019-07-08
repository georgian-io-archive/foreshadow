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
    """Return Foreshadow logger instance."""
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
            _stream_target = _sys.stdout
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
    log_fn = getattr(_this_module, level, None)
    if log_fn is None:
        raise ValueError("level: '{}' is not a valid logging level.")
    return log_fn


class GuiEvent(object):  # TODO use serializer interface to serialize
    def __init__(self, object=None, method=None, details=None):
        from datetime import datetime

        timestamp = datetime.now()
        self.time = timestamp.strftime("%Y%m%dT%H:%M:%S:%f")
        self.object = object
        self.method = method
        self.details = details


class AsyncWrite(_threading.Thread):
    def __init__(
        self, buffer_size=100, outfile="gui_data.txt", overwrite=True
    ):
        super(AsyncWrite, self).__init__()
        self._buffer_size = buffer_size
        self.outfile = outfile
        self.first_write = overwrite
        self.buffer = []

    @property
    def buffer_size(self):
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
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

    @staticmethod
    def _format_for_gui(object, method, out):
        gui_event = GuiEvent(object=object, method=method, details=out)
        return str(gui_event.__dict__)  # TODO use serializer to serialize

    def write(self):
        log_fn = _get_log_fn("debug")
        log_fn("writing gui buffer with: %d msgs to file" % len(self.buffer))
        io_type = "w" if self.first_write else "a"
        with open(self.outfile, io_type) as outfile:
            for msg in self.buffer:
                outfile.write(msg)
                outfile.write("\n")
        self.first_write = False
        self.buffer = []

    def __call__(self, object, method, out, force=False):
        msg = self._format_for_gui(object, method, out)
        # perform write operation to some nosql database or file
        if len(self.buffer) >= self.buffer_size - 1 or force:  # -1 as append.
            self.buffer.append(msg)
            self.write()
        else:
            self.buffer.append(msg)


gui_fn = AsyncWrite()


def set_level(level_name):
    level = LEVELS[level_name]
    get_logger().setLevel(level)


def __log(level, msg, *args, **kwargs):
    log_fn = getattr(get_logger(), level, None)
    if log_fn is None:
        raise NotImplementedError(
            "please implement the logging function "
            "for level: '{}'".format(level)
        )
    log_fn(msg, *args, **kwargs)


def __wrap_log(func, level):
    def wrapped_func(*args, **kwargs):
        return func(level, *args, **kwargs)

    wrapped_func.__name__ = level
    return wrapped_func


for level in LEVELS:
    log = __wrap_log(__log, level)
    setattr(_this_module, level, log)


def log_and_gui(level, object, method, msg, *args, **kwargs):
    log_fn = _get_log_fn(level)
    log_fn(msg, *args, **kwargs)
    gui_fn(object=object, method=method, out=msg)


@_atexit.register
def sync_gui():
    log_fn = _get_log_fn("info")
    log_fn("asynchronous gui thread still writing. Now waiting on it.")
    gui_fn.join()
    log_fn = _get_log_fn("debug")
    log_fn("asynchronous gui thread finished.")
