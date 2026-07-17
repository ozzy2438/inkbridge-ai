"""TrOCR Fine-tuning Pipeline.

Fine-tunes TrOCR on domain-specific handwriting data:
- SMHD (student messy handwriting)
- GNHK (camera-captured handwriting)
- Custom augmented data

Key features:
- Writer-independent train/test split
- Progressive training (curriculum learning)
- Confidence calibration during training
- Early stopping on gold validation set
"""

from dataclasses import dataclass
from typing import Any

import structlog
import torch
from torch.utils.data import DataLoader

logger = structlog.get_logger()


@dataclass
class TrOCRTrainingConfig:
    """Configuration for TrOCR fine-tuning."""

    # Model
    base_model: str = "microsoft/trocr-base-handwritten"
    output_dir: str = "./checkpoints/trocr-finetuned"

    # Training hyperparameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.0005
    num_epochs: int = 15
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler_type: str = "cosine"

    # Generation
    max_length: int = 128
    beam_width: int = 4

    # Data
    train_data_dir: str = "./data/processed/train"
    val_data_dir: str = "./data/processed/val"
    test_data_dir: str = "./data/processed/test"

    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5

    # Training settings
    fp16: bool = True
    dataloader_num_workers: int = 4
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    early_stopping_patience: int = 5

    # Logging
    logging_steps: int = 100
    report_to: str = "tensorboard"
    run_name: str | None = None


class TrOCRTrainer:
    """Fine-tunes TrOCR for domain-specific handwriting recognition.

    Training pipeline:
    1. Load base TrOCR model (pretrained on IAM)
    2. Prepare writer-independent data splits
    3. Apply augmentations (blur, noise, rotation)
    4. Train with early stopping on validation CER
    5. Evaluate on held-out gold test set
    6. Export best checkpoint
    """

    def __init__(self, config: TrOCRTrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any = None
        self.processor: Any = None
        self.best_cer = float("inf")

    def setup(self):
        """Initialize model, processor, and datasets."""
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )

        logger.info("trainer.setup", base_model=self.config.base_model)

        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(self.config.base_model)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.base_model)

        # Configure model for training
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = self.config.max_length
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = self.config.beam_width

        logger.info("trainer.setup.complete")

    def prepare_datasets(self):
        """Prepare training and validation datasets."""
        from src.data.augmentation import HandwritingAugmentation
        from src.data.datasets import HandwritingDataset

        augmentation = None
        if self.config.use_augmentation:
            augmentation = HandwritingAugmentation(probability=self.config.augmentation_prob)

        self.train_dataset = HandwritingDataset(
            data_dir=self.config.train_data_dir,
            processor=self.processor,
            max_length=self.config.max_length,
            augmentation=augmentation,
        )

        self.val_dataset = HandwritingDataset(
            data_dir=self.config.val_data_dir,
            processor=self.processor,
            max_length=self.config.max_length,
            augmentation=None,  # No augmentation for validation
        )

        logger.info(
            "trainer.datasets_prepared",
            train_size=len(self.train_dataset),
            val_size=len(self.val_dataset),
        )

    def train(self):
        """Run the fine-tuning training loop."""
        import evaluate
        from transformers import (
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            default_data_collator,
        )

        cer_metric = evaluate.load("cer")

        def compute_metrics(pred):
            """Compute CER metric during evaluation."""
            labels_ids = pred.label_ids
            pred_ids = pred.predictions

            # Replace -100 with pad token
            labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            cer = cer_metric.compute(predictions=pred_str, references=label_str)

            return {"cer": cer}

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            fp16=self.config.fp16,
            output_dir=self.config.output_dir,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            report_to=self.config.report_to,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            lr_scheduler_type=self.config.lr_scheduler_type,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            dataloader_num_workers=self.config.dataloader_num_workers,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            run_name=self.config.run_name,
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.processor.feature_extractor,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=default_data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)
            ],
        )

        logger.info("trainer.starting")

        # Train
        train_result = trainer.train()

        # Save best model
        trainer.save_model(self.config.output_dir)
        self.processor.save_pretrained(self.config.output_dir)

        logger.info(
            "trainer.complete",
            train_loss=train_result.training_loss,
            best_cer=trainer.state.best_metric,
        )

        return train_result

    def evaluate(self, test_data_dir: str | None = None):
        """Evaluate model on test set."""
        from src.data.datasets import HandwritingDataset
        from src.evaluation.metrics import compute_full_metrics

        test_dir = test_data_dir or self.config.test_data_dir

        test_dataset = HandwritingDataset(
            data_dir=test_dir,
            processor=self.processor,
            max_length=self.config.max_length,
        )

        logger.info("trainer.evaluating", test_size=len(test_dataset))

        # Run predictions
        predictions = []
        references = []

        self.model.eval()
        dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)

                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.config.max_length,
                    num_beams=self.config.beam_width,
                )

                pred_str = self.processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                predictions.extend(pred_str)

                labels = batch["labels"]
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                label_str = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                references.extend(label_str)

        # Compute metrics
        metrics = compute_full_metrics(predictions, references)

        logger.info("trainer.evaluation.complete", **metrics)

        return metrics
