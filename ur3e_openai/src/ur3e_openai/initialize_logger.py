import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter

GREEN = 21
logging.addLevelName(GREEN, "GREEN")
def green(self, message, *args, **kws):
    if self.isEnabledFor(GREEN):
        # Yes, logger takes its '*args' as 'args'.
        self._log(GREEN, message, args, **kws) 
logging.Logger.green = green

class CustomFormatter(logging.Formatter):

    cyan = '\033[36m'
    green = '\033[32m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold_red = '\033[1;31m'
    reset = '\033[0m'
    format = '%(levelname)s-%(filename)s:%(lineno)s %(message)s'

    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: format,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
        GREEN: green + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def initialize_logger(logging_level=logging.INFO, filename=None, save_log=True, log_tag="ur3e_openai"):
    logger = getLogger(log_tag)
    logger.setLevel(logging_level)

    handler_format = Formatter('%(levelname)s-%(filename)s:%(lineno)s %(message)s')
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(CustomFormatter())

    if save_log:
        if filename is not None:
            file_handler = FileHandler(filename, 'a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(handler_format)
        else:
            save_log = False

    if len(logger.handlers) == 0:
        logger.addHandler(stream_handler)
        if save_log:
            logger.addHandler(file_handler)
    else:
        # Overwrite logging setting
        logger.handlers[0] = stream_handler
        if save_log:
            logger.handlers[1].close()
            logger.handlers[1] = file_handler

    logger.propagate = False

    return logger
