import json
from pydantic import BaseModel, Field
from typing import Optional


class DirectoriesConfig(BaseModel):
    input_dir: str = Field(
        ..., description="Directory to load train and validation datasets"
    )
    clean_train_dir_before_training: Optional[bool] = Field(
        default=True,
        description="Whether to clean the training directory before training",
    )
    train_dir: str = Field(..., description="Directory to save the training files")
    training_curve_path: str = Field(
        ..., description="Path to save the loss and metrics curves after training"
    )
    best_model_path: str = Field(
        ..., description="Path to save the best model during training"
    )


class ModelConfig(BaseModel):
    model_name: str = Field(
        ..., description="The model to use for image classification"
    )
    nb_layers_to_freeze: Optional[int] = Field(
        default=None,
        ge=0,
        le=12,
        description="Number of initial ViT encoder layers to freeze (1-12).",
    )


class TrainingHyperparams(BaseModel):
    enable_gpu: Optional[bool] = Field(
        default=False, description="Whether to use GPU if available"
    )
    learning_rate: float = Field(..., description="Learning rate for the optimizer")
    batch_size: int = Field(..., description="Batch size for training")
    num_train_epochs: int = Field(..., description="Number of training epochs")


class TrainingConfig(BaseModel):
    directories_config: DirectoriesConfig
    model_params: ModelConfig
    training_config: TrainingHyperparams


def load_training_config(config_path: str) -> TrainingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return TrainingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")
