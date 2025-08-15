from src.config_loaders.training_config_loader import TrainingConfig
from src.base_pipeline import BasePipeline
from colorama import Fore, Style
from datasets import load_from_disk
from src.modeling.model_builder import ModelBuilder
from transformers import TrainingArguments, Trainer
from src.utils.schema import BatchSchema, MetricsSchema
from src.evaluators.accuracy import compute_accuracy
from src.utils.utils_toolbox import (
    clean_checkpoints,
    plot_training_and_validation_curves,
)


class TrainingPipeline(BasePipeline):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    def run(self) -> None:

        print(f"{Fore.GREEN}Starting training pipeline...{Style.RESET_ALL}")

        # Load train and validation sets
        print(f"{Fore.YELLOW}Loading data from specified paths...{Style.RESET_ALL}")
        input_dir = self.config.input_dir
        try:
            train_dataset = load_from_disk(f"{input_dir}/train")
            val_dataset = load_from_disk(f"{input_dir}/val")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{Fore.RED}Could not find train or val datasets at {input_dir}{Style.RESET_ALL}"
            )

        # Create label to class ID mapping
        train_labels = train_dataset.features["labels"].names
        val_labels = val_dataset.features["labels"].names
        assert set(train_labels) == set(
            val_labels
        ), f"{Fore.RED}Train and validation sets have different labels{Style.RESET_ALL}"
        label2id, id2label = dict(), dict()
        for i, label in enumerate(train_labels):
            label2id[label] = i
            id2label[i] = label

        # Build the model
        print(f"{Fore.YELLOW}Creating Model...{Style.RESET_ALL}")
        model_builder = ModelBuilder(
            model_name=self.config.model_name,
            num_labels=len(train_labels),
            id2label=id2label,
            label2id=label2id,
            enable_gpu=self.config.enable_gpu,
            nb_layers_to_freeze=self.config.nb_layers_to_freeze,
        )
        _transforms, model, device = model_builder.initialize()
        model = model.to(device)

        # Image preprocessing
        def apply_transforms(batch: dict) -> dict:
            batch[BatchSchema.PIXEL_VALUES] = [
                _transforms(img.convert("RGB")) for img in batch["image"]
            ]
            del batch["image"]
            return batch

        train_dataset = train_dataset.with_transform(apply_transforms)
        val_dataset = val_dataset.with_transform(apply_transforms)

        # Training arguments
        # [MEDIUM]: add training params in config
        args = TrainingArguments(
            output_dir=self.config.train_dir,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            load_best_model_at_end=True,
            metric_for_best_model=MetricsSchema.ACCURACY,
            greater_is_better=True,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_accuracy,
        )

        # Run training loop
        if self.config.clean_train_dir_before_training:
            clean_checkpoints(train_dir=self.config.train_dir)

        trainer.train()

        # Save the training and validation curves (loss and accuracy)
        training_logs = trainer.state.log_history
        train_losses = [log["loss"] for log in training_logs if "loss" in log]
        val_losses = [log["eval_loss"] for log in training_logs if "eval_loss" in log]
        val_accuracies = [
            log[f"eval_{MetricsSchema.ACCURACY}"]
            for log in training_logs
            if f"eval_{MetricsSchema.ACCURACY}" in log
        ]
        plot_training_and_validation_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics=val_accuracies,
            save_path=self.config.training_curve_path,
        )

        # Save the best model for Testing and Inference
        trainer.save_model(self.config.best_model_path)

        print(f"{Fore.GREEN}Training pipeline completed successfully!{Style.RESET_ALL}")
