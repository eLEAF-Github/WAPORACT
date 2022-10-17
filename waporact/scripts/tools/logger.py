"""
small script for setting the package wide specific logger
"""
import logging


def format_root_logger(
    logging_level=None,
    message_format: str = "%(asctime)s    %(name)s    %(lineno)d    %(levelname)s  %(message)s",
    # message_format: str = "[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s",
    # dateformat: str = "%d/%m/%Y %I:%M:%S %p",
    dateformat: str = "%m-%d %H:%M:%S",
    log_to_console: bool = True,
    log_file: str = None,
):
    root_logger = logging.getLogger()

    # check for and as needed create main logger
    if logging_level is None:
        logging_level = logging.INFO

    root_logger.setLevel(logging_level)

    formatter = logging.Formatter(
        message_format,
        datefmt=dateformat,
    )

    if log_to_console:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging_level)
        consoleHandler.setFormatter(formatter)
        root_logger.addHandler(consoleHandler)

    if isinstance(log_file, str):
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setLevel(logging_level)
        fileHandler.setFormatter(formatter)
        root_logger.addHandler(fileHandler)

    return 0
