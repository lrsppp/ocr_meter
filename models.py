from pydantic import BaseModel
from typing import Any


class ImageModel(BaseModel):
    image_data: Any
