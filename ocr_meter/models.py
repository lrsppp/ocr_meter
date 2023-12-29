from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel


class ImageModel(BaseModel):
    image_data: Any

    @classmethod
    def from_png(cls, path: (Path | str)):
        data = cv2.imread(path)
        return ImageModel(image_data=data.tolist())


class LogConfig(BaseModel):
    LOGGER_NAME: str = "ocr_meter"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"

    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict[str, Any] = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict[str, str] = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers: dict[str, Any] = {
        LOGGER_NAME: {"handlers": ["default"], "level": LOG_LEVEL},
    }
