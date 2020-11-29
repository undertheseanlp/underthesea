###########################################################
# Initialize
###########################################################
import torch
import logging.config

# global variable: device for torch
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "underthesea": {"handlers": ["console"], "level": "INFO", "propagate": False}
        },
    }
)

logger = logging.getLogger("underthesea")
