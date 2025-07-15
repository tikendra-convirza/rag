import logging
import sys
import queue
import threading

from logging.handlers import QueueHandler, QueueListener

_log_queue = queue.Queue(-1)
_listener = None

def setup_logger(name: str = "rag", level: int = logging.INFO) -> logging.Logger:
    global _listener
    logger = logging.getLogger(name)
    if not any(isinstance(h, QueueHandler) for h in logger.handlers):
        # Create a handler for the queue
        qh = QueueHandler(_log_queue)
        logger.addHandler(qh)
        logger.setLevel(level)
        # Only start one listener for the process
        if _listener is None:
            stream_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            stream_handler.setFormatter(formatter)
            _listener = QueueListener(_log_queue, stream_handler)
            _listener.start()
    return logger

# Usage:
# from src.logger import setup_logger
# logger = setup_logger(__name__)
