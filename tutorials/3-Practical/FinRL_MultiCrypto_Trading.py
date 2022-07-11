import logging

logger = logging.getLogger()
logger.setLevel("DEBUG")
file_handler = logging.FileHandler("./log.txt", mode='a', encoding="utf-8")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(logging.Formatter(fmt="%(lineno)s---%(asctime)s---%(message)s"))
logger.addHandler(file_handler)

# please use logging.info() instead print()
