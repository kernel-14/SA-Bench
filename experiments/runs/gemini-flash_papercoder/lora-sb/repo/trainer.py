import torch
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer
from transformers.trainer_utils import TrainOutput
from datasets import Dataset # Assuming HuggingFace Dataset type
from typing import Callable, Dict, Any

from lora_sb import LoRASBModel # Import the custom LoRA-SB model

class LoRASBTrainer(Trainer):
    """
    A specialized Trainer class for fine-tuning models adapted with LoRA-SB.
    This class leverages HuggingFace's Trainer capabilities while working
    with the custom LoRASBModel.
    """

    def __init__(
        self,
        model: LoRASBModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        tokenizer: PreTrainedTokenizer,
        data_collator: Callable,
        # compute_metrics: Optional[Callable] = None, # Optionally add compute_metrics if needed later for Trainer's internal evaluation
        **kwargs,
    ):
        """
        Initializes the LoRASBTrainer.

        Args:
            model (LoRASBModel): The LoRA-SB adapted model to be trained.
            args (TrainingArguments): Arguments for the Trainer.
            train_dataset (Dataset): The training dataset.
            eval_dataset (Optional[Dataset]): The evaluation dataset. Can be None if no evaluation is performed during training.
            tokenizer (PreTrainedTokenizer): The tokenizer used for data processing.
            data_collator (Callable): The function used to form batches from dataset elements.
            **kwargs: Additional keyword arguments to pass to the base Trainer.
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics, # Pass compute_metrics if provided
            **kwargs,
        )

    def train(self) -> TrainOutput:
        """
        Runs the training loop.

        This method delegates directly to the parent `transformers.Trainer.train()` method.
        The `transformers.Trainer` will automatically identify trainable parameters
        (which are only the `R` matrices in `LoRASBModel` due to its design)
        and handle the optimization process.

        Returns:
            TrainOutput: An object containing information about the training run,
                         including the state of the model and optimizer at the end.
        """
        return super().train()

