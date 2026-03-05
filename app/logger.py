import logging
import json
import sys
from datetime import datetime


def setup_logger():
    logger = logging.getLogger("fraud_system")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger = setup_logger()


def log_event(event_type: str, payload: dict):
    log_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        **payload,
    }
    logger.info(json.dumps(log_record))