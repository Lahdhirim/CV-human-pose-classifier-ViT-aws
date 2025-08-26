from pydantic import BaseModel
from pydantic import HttpUrl


class ImageDataInput(BaseModel):
    url: HttpUrl


class ImageDataOutput(BaseModel):
    top_labels: list[str]
    top_probs: list[float]
    prediction_time: int
