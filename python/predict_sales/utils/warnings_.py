import logging
from contextlib import ContextDecorator


def set_warnings_handlers_from(logger: logging.Logger):
    """
    Copy handlers from specified Logger instance to "py.warnings" logger.
    """
    wlogger = logging.getLogger("py.warnings")
    wlogger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        wlogger.addHandler(h)


class warnings_to_log(ContextDecorator):
    """
    Context manager to redirect warnings to logging module.
    Accepts optional category or string arguments, (subclasses of Warning or
    strings to match in the warning message),
    and an action argument to filter the log records of warnings belonging
    to these categories.
    :param categories: Warning categories to apply the specified action.
        Warnings not belonging to these categories will be logged as level WARNING.
    :param action: default "debug".
        "debug" - change level of log record to DEBUG (from WARNING)
        "ignore" - filter out these warnings. (won't be passed on to any logger)
        "log" - handled by loggers (as level WARNING)
    """
    def __init__(self, *categories, action="debug"):
        self.Filter = PyWarningsFilter(*categories, action=action)

    def __enter__(self):
        logging.getLogger("py.warnings").addFilter(self.Filter)
        logging.captureWarnings(True)

    def __exit__(self, *exc):
        logging.getLogger("py.warnings").removeFilter(self.Filter)
        logging.captureWarnings(False)
        return False


class PyWarningsFilter:
    def __init__(self, *categories, action="debug"):
        self.categories = categories
        assert action in {"debug", "ignore", "log"}
        self.action = action

    def filter(self, record):
        for cat in self.categories:
            if isinstance(cat, Warning):
                cat = " {}: ".format(cat.__name__)
            if cat in record.getMessage():
                if self.action == "debug":
                    record.levelno = logging.DEBUG
                    record.levelname = 'DEBUG'
                    return True
                if self.action == "ignore":
                    return False
                if self.action == "log":
                    return True
        return True
