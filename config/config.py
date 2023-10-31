import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

# Logs path
logs_path = os.path.join((Path(__file__).parent.parent), "logs")

# Create logs folder
os.makedirs(logs_path, exist_ok=True)

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
console_handler = RichHandler(markup=True)
console_handler.setLevel(logging.INFO)

info_handler = RotatingFileHandler(
    filename=Path(logs_path, "info.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
info_handler.setLevel(logging.INFO)

error_handler = RotatingFileHandler(
    filename=Path(logs_path, "error.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
error_handler.setLevel(logging.ERROR)

# Create formatters
minimal_formatter = logging.Formatter(fmt="%(message)s")
detailed_formatter = logging.Formatter(fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n")

# Hook it all up
console_handler.setFormatter(fmt=minimal_formatter)
info_handler.setFormatter(fmt=detailed_formatter)
error_handler.setFormatter(fmt=detailed_formatter)
logger.addHandler(hdlr=console_handler)
logger.addHandler(hdlr=info_handler)
logger.addHandler(hdlr=error_handler)
