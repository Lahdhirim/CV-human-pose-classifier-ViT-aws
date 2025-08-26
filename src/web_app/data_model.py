from pydantic import BaseModel


class ImageDataOutput(BaseModel):
    top_labels: list[str]
    top_probs: list[float]
    prediction_time: int
