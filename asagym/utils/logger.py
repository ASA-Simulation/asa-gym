import logging
import pathlib


def new_logger(
    file: pathlib.Path, name: str, level: str, print: bool = False
) -> logging.Logger:
    level_value = logging.getLevelName(level.upper())
    logging.basicConfig(
        filename=file,
        level=level_value,
        format="%(asctime)s <%(levelname)s> [%(name)s] %(message)s",
    )
    logger = logging.getLogger(name)
    if print:
        # print to stderr also (optional)
        logger.addHandler(logging.StreamHandler())
    return logger


def fork_logger(name: str, logger: logging.Logger) -> logging.Logger:
    return logger.getChild(name)
