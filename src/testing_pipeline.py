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
from src.utils.utils_toolbox import save_classification_results_to_excel
from src.evaluators.classification_metrics import compute_results
from src.aws_services.s3_service import S3Manager


class TestingPipeline(BasePipeline):
    def __init__(self, config: TestingConfig):
        super().__init__(config)

    def run(self):

        print(f"{Fore.GREEN}Starting testing pipeline...{Style.RESET_ALL}")

        # Load test data
        print(f"{Fore.YELLOW}Loading test data from specified path...{Style.RESET_ALL}")
        input_dir = self.config.input_dir
        try:
            test_dataset = load_from_disk(f"{input_dir}")
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
        metrics_df, cm_df, per_class_acc_df = compute_results(
            labels_list, preds, labels
        )
        save_classification_results_to_excel(
            metrics_df=metrics_df,
            cm_df=cm_df,
            per_class_acc_df=per_class_acc_df,
            output_file=self.config.metrics_output_file,
        )

        # Push the model to S3 Bucket if it reaches the required performances
        metrics = metrics_df.iloc[0].to_dict()
        if self.config.push_model_s3:
            push_model_s3_config = self.config.push_model_s3
            if push_model_s3_config.enabled:
                if len(push_model_s3_config.conditions) > 0:
                    print(
                        f"{Fore.YELLOW}Verifying conditions to push model to S3 bucket...{Style.RESET_ALL}"
                    )
                    model_valid = True
                    for condition in push_model_s3_config.conditions:
                        assert (
                            condition.metric in metrics
                        ), f"Metric {condition.metric} not found in the metrics dictionary {list(metrics.keys())}."
                        if metrics[condition.metric] < condition.threshold:
                            model_valid = False
                            print(
                                f"{Fore.RED}Metric {condition.metric} ({metrics[condition.metric]}) does not meet the threshold of {condition.threshold}.\nModel will not be pushed to S3 bucket...{Style.RESET_ALL}"
                            )
                            break

                    if model_valid:
                        print(
                            f"{Fore.GREEN}Model validation passed. Pushing model to S3 bucket...{Style.RESET_ALL}"
                        )
                        s3_manager = S3Manager(
                            bucket_name=push_model_s3_config.bucket_name
                        )
                        s3_manager.create_bucket_if_not_exists()
                        s3_manager.upload_directory(
                            local_directory_path=self.config.trained_model_path,
                            s3_prefix=push_model_s3_config.prefix,
                        )

                else:
                    print(
                        f"{Fore.RED}No conditions specified for pushing model to S3 bucket. Skipping the push operation...{Style.RESET_ALL}"
                    )
            else:
                print(
                    f"{Fore.RED}Pushing model to S3 bucket is disabled. Skipping the push operation...{Style.RESET_ALL}"
                )
        else:
            print(
                f"{Fore.YELLOW}Pushing model to S3 bucket is not configured. Skipping the push operation...{Style.RESET_ALL}"
            )

        print(f"{Fore.GREEN}Testing pipeline completed successfully!{Style.RESET_ALL}")
