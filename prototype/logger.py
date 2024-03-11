import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from config import AppConfigSettings


def setup_logging(
    config: AppConfigSettings, job_name: Optional[str] = None
) -> logging.Logger:
    filename = job_name or config.app.app_name

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = level_map.get(config.app.log_level.lower(), logging.INFO)
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = TimedRotatingFileHandler(
        f"{filename}.log", when="midnight", interval=1, backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
