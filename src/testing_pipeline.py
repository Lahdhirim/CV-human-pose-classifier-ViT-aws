import os
import pickle
import torch
from tqdm import tqdm
from colorama import Fore, Style
import pandas as pd

from transformers import AutoModelForImageClassification
from datasets import load_from_disk

from src.config_loaders.testing_config_loader import TestingConfig
from src.base_pipeline import BasePipeline
from src.utils.schema import BatchSchema, SavingSchema
from src.evaluators.classification_metrics import compute_results


class TestingPipeline(BasePipeline):
    def __init__(self, config: TestingConfig):
        super().__init__(config)

    def run(self):

        print(f"{Fore.GREEN}Starting testing pipeline...{Style.RESET_ALL}")

        # Load test data
        print(f"{Fore.YELLOW}Loading test data from specified path...{Style.RESET_ALL}")
        input_dir = self.config.input_dir
        try:
            test_dataset = (
                load_from_disk(f"{input_dir}").shuffle(seed=42).select(range(50))
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{Fore.RED}Could not find test dataset at {input_dir}{Style.RESET_ALL}"
            )

        # Load the model
        print(
            f"{Fore.YELLOW}Loading trained model from {self.config.trained_model_path}{Style.RESET_ALL}"
        )
        labels = test_dataset.features["labels"].names
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for label, i in label2id.items()}
        model = AutoModelForImageClassification.from_pretrained(
            self.config.trained_model_path,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        model.eval()

        # Load _transforms
        transforms_file_path = os.path.join(
            self.config.trained_model_path, SavingSchema.TRANSFORMS_PKL
        )
        try:
            with open(transforms_file_path, "rb") as f:
                _transforms = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{Fore.RED}Could not find transforms at {transforms_file_path}{Style.RESET_ALL}"
            )

        # Apply transformations on test set
        def apply_transforms(batch: dict) -> dict:
            batch[BatchSchema.PIXEL_VALUES] = [
                _transforms(img.convert("RGB")) for img in batch["image"]
            ]
            del batch["image"]
            return batch

        test_dataset = test_dataset.with_transform(apply_transforms)

        # Predictions
        print(f"{Fore.YELLOW}Making predictions on test set...{Style.RESET_ALL}")
        preds, labels_list = [], []
        for sample in tqdm(test_dataset, desc="Predicting", unit="sample"):

            pixel_values = sample[BatchSchema.PIXEL_VALUES].unsqueeze(0)
            true_label = sample["labels"]

            with torch.no_grad():
                outputs = model(pixel_values)
                pred_label_id = outputs.logits.argmax(-1).item()

            preds.append(pred_label_id)
            labels_list.append(true_label)

        # Compute classifications metrics and save results into Excel file
        print(f"{Fore.YELLOW}Calculating classifications metrics...{Style.RESET_ALL}")
        metrics_df, cm_df = compute_results(labels_list, preds, labels)
        with pd.ExcelWriter(
            self.config.metrics_output_file, engine="openpyxl"
        ) as writer:

            metrics_df.to_excel(writer, sheet_name="metrics", index=False)

            cm_df.index.name = "Ground Truth ↓ / Prediction →"
            cm_styled = (
                cm_df.style.background_gradient(cmap="Blues")
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("font-weight", "bold"),
                                ("text-align", "center"),
                            ],
                        }
                    ]
                )
                .set_properties(**{"text-align": "center"})
            )
            cm_styled.to_excel(writer, sheet_name="confusion_matrix")

        print(
            f"{Fore.CYAN}Evaluation results saved to {self.config.metrics_output_file}{Style.RESET_ALL}"
        )

        print(f"{Fore.GREEN}Testing pipeline completed successfully!{Style.RESET_ALL}")
