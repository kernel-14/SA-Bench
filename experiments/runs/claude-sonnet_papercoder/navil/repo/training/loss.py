## training/loss.py
"""Next-Token-Prediction (NTP) loss for NaViL training.

This module implements the unified training objective used across all three
NaViL training stages (S1.1, S1.2, S2). The paper explicitly states that
native MLLMs use a single Next-Token-Prediction objective applied uniformly
to image-caption pairs and instruction-following data alike (Section 3.1):

    "All the models are trained on web-scale, noisy image-caption pair data
    with Next-Token-Prediction (NTP) and an image captioning task."

The loss is a standard autoregressive cross-entropy with shift-by-one and
ignore_index=-100 masking. The masking convention is shared across the entire
codebase (noted in "Shared Knowledge"):

    labels[i] == -100  →  position i is excluded from loss computation

Three categories of positions are masked to -100 by the dataset/collation:
    1. Padding tokens (right-padded sequences in a batch)
    2. Image token positions (visual patches should not be predicted)
    3. Prompt/instruction prefix tokens in S2 (only assistant turns contribute)

Config alignment (configs/navil_2b.yaml):
    training.optimizer.gradient_accumulation_steps: 1
        → loss is not divided here; accelerate handles accumulation in trainer
    training.precision: "bfloat16"
        → logits arrive in bfloat16; CrossEntropyLoss internally upcasts to
          float32 for numerical stability (standard PyTorch behavior)

Dependencies:
    - torch: tensor operations
    - torch.nn: CrossEntropyLoss, Module base class
    No internal project dependencies — this is a leaf module.
"""

import torch
import torch.nn as nn


class NTPLoss(nn.Module):
    """Next-Token-Prediction loss for autoregressive multimodal training.

    Wraps ``torch.nn.CrossEntropyLoss`` with the standard shift-by-one
    operation required for autoregressive language modeling. The loss is
    computed only over non-masked positions (label != ignore_index).

    The shift-by-one operation:
        - ``shift_logits = logits[:, :-1, :]``  — predictions at positions 0..L-2
        - ``shift_labels = labels[:, 1:]``       — targets at positions 1..L-1

    This means the model at position ``i`` predicts the token at position
    ``i+1``, which is the standard autoregressive NTP formulation.

    Args:
        ignore_index: Token label value to exclude from loss computation.
                      Defaults to ``-100``, which is the PyTorch convention
                      and the value used throughout the NaViL codebase for
                      padding, image token positions, and prompt prefixes.

    Attributes:
        ignore_index: Stored ignore index value.
        _ce_loss:     Internal ``torch.nn.CrossEntropyLoss`` instance with
                      ``reduction='mean'`` and the configured ``ignore_index``.
                      Created once in ``__init__`` and reused across all
                      forward calls (no learnable parameters).

    Example::

        loss_fn = NTPLoss(ignore_index=-100)

        # logits from MoELLM.lm_head: (B, L, vocab_size)
        logits = torch.randn(2, 128, 32000, dtype=torch.bfloat16)

        # labels with -100 at image positions and padding
        labels = torch.randint(0, 32000, (2, 128), dtype=torch.long)
        labels[:, :20] = -100   # image block masked
        labels[:, 100:] = -100  # padding masked

        loss = loss_fn(logits, labels)
        # loss: scalar float tensor
        loss.backward()
    """

    def __init__(self, ignore_index: int = -100) -> None:
        """Initialise NTPLoss with a single CrossEntropyLoss instance.

        Args:
            ignore_index: Label value to exclude from loss computation.
                          Defaults to ``-100`` (PyTorch convention, shared
                          across the NaViL codebase).
        """
        super().__init__()

        self.ignore_index: int = ignore_index

        # Create the CrossEntropyLoss instance once.
        # reduction='mean': average over all non-ignored positions in the batch.
        # This is the standard choice for language model training — it gives
        # a per-token loss that is comparable across batches with different
        # numbers of valid (non-masked) tokens.
        self._ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean NTP loss over all non-masked token positions.

        Applies the shift-by-one operation to align predictions with targets,
        then flattens the batch and sequence dimensions for cross-entropy
        computation.

        Data flow::

            logits: (B, L, vocab_size)
            labels: (B, L)

            shift_logits = logits[:, :-1, :]        → (B, L-1, vocab_size)
            shift_labels = labels[:, 1:]            → (B, L-1)

            flat_logits  = shift_logits.reshape(-1, vocab_size)  → (B*(L-1), vocab_size)
            flat_labels  = shift_labels.reshape(-1)              → (B*(L-1),)

            loss = CrossEntropyLoss(flat_logits, flat_labels)    → scalar

        The ``CrossEntropyLoss`` internally upcasts logits to float32 for
        numerical stability, even when the input is bfloat16. This is the
        correct behavior for mixed-precision training.

        Args:
            logits: Raw (pre-softmax) logits from ``MoELLM.lm_head``.
                    Shape: ``(B, L, vocab_size)`` where:
                        B = batch size
                        L = sequence length (including image tokens)
                        vocab_size = tokenizer vocabulary size (inferred at runtime)
                    Dtype: typically ``torch.bfloat16`` during training
                    (from config: ``training.precision: "bfloat16"``).
            labels: Integer token IDs with ``ignore_index`` (default -100) at
                    masked positions. Shape: ``(B, L)``, dtype ``torch.long``.
                    Masked positions include:
                        - Padding tokens (right-padded to batch max length)
                        - Image token positions (visual patches not predicted)
                        - Prompt/instruction prefix tokens in S2

        Returns:
            Scalar float tensor representing the mean NTP loss over all
            valid (non-ignored) token positions in the batch. The tensor
            is on the same device as ``logits`` and has dtype ``torch.float32``
            (CrossEntropyLoss always returns float32 regardless of input dtype).

        Raises:
            ValueError: If ``logits`` does not have exactly 3 dimensions
                        ``(B, L, vocab_size)``.
            ValueError: If ``labels`` does not have exactly 2 dimensions
                        ``(B, L)``.
            ValueError: If the batch size or sequence length of ``logits``
                        and ``labels`` do not match.

        Note:
            If all label positions in the batch are masked (all ``-100``),
            ``CrossEntropyLoss`` with ``reduction='mean'`` returns ``nan``
            (division by zero over zero valid positions). The caller
            (``NaViLTrainer.train_step``) should guard against this edge
            case if it can arise in practice (e.g., by filtering empty batches
            in the data pipeline).

        Example::

            loss_fn = NTPLoss(ignore_index=-100)

            # Typical training step
            logits = model_output.logits          # (2, 512, 32000), bfloat16
            labels = batch["labels"]              # (2, 512), long

            loss = loss_fn(logits, labels)        # scalar, float32
            accelerator.backward(loss)
        """
        # ------------------------------------------------------------------ #
        # Input validation                                                     #
        # ------------------------------------------------------------------ #
        if logits.dim() != 3:
            raise ValueError(
                f"logits must have exactly 3 dimensions (B, L, vocab_size), "
                f"got shape {tuple(logits.shape)} with {logits.dim()} dimensions."
            )

        if labels.dim() != 2:
            raise ValueError(
                f"labels must have exactly 2 dimensions (B, L), "
                f"got shape {tuple(labels.shape)} with {labels.dim()} dimensions."
            )

        B_logits: int
        L_logits: int
        vocab_size: int
        B_logits, L_logits, vocab_size = logits.shape

        B_labels: int
        L_labels: int
        B_labels, L_labels = labels.shape

        if B_logits != B_labels:
            raise ValueError(
                f"Batch size mismatch: logits has B={B_logits}, "
                f"labels has B={B_labels}. "
                "Ensure logits and labels come from the same forward pass."
            )

        if L_logits != L_labels:
            raise ValueError(
                f"Sequence length mismatch: logits has L={L_logits}, "
                f"labels has L={L_labels}. "
                "Ensure logits and labels are constructed from the same "
                "input_ids tensor without independent truncation."
            )

        # ------------------------------------------------------------------ #
        # Shift-by-one for autoregressive NTP                                 #
        # ------------------------------------------------------------------ #
        # The model at position i predicts the token at position i+1.
        # shift_logits[b, i, :] = logits[b, i, :]     → predicts labels[b, i+1]
        # shift_labels[b, i]    = labels[b, i+1]       → target for logits[b, i]
        #
        # After shifting, both tensors have sequence length L-1.
        # The last logit position (predicting beyond the sequence) is dropped.
        # The first label position (which has no preceding context) is dropped.
        shift_logits: torch.Tensor = logits[:, :-1, :].contiguous()
        # shape: (B, L-1, vocab_size)

        shift_labels: torch.Tensor = labels[:, 1:].contiguous()
        # shape: (B, L-1)

        # ------------------------------------------------------------------ #
        # Flatten batch and sequence dimensions for CrossEntropyLoss          #
        # ------------------------------------------------------------------ #
        # CrossEntropyLoss expects:
        #   input:  (N, C) where N = number of samples, C = number of classes
        #   target: (N,)
        #
        # We flatten (B, L-1) → (B*(L-1),) for both tensors.
        # .reshape(-1, vocab_size) is equivalent to .view(-1, vocab_size) here
        # since .contiguous() was called above, but .reshape is safer.
        flat_logits: torch.Tensor = shift_logits.reshape(-1, vocab_size)
        # shape: (B*(L-1), vocab_size)

        flat_labels: torch.Tensor = shift_labels.reshape(-1)
        # shape: (B*(L-1),)

        # ------------------------------------------------------------------ #
        # Compute cross-entropy loss                                           #
        # ------------------------------------------------------------------ #
        # CrossEntropyLoss internally:
        #   1. Upcasts logits to float32 (numerical stability for bfloat16 input)
        #   2. Applies log-softmax over the vocab_size dimension
        #   3. Computes negative log-likelihood at the target positions
        #   4. Excludes positions where flat_labels == ignore_index (-100)
        #   5. Returns mean over all non-ignored positions
        loss: torch.Tensor = self._ce_loss(flat_logits, flat_labels)
        # shape: scalar (0-dimensional tensor)
        # dtype: torch.float32 (CrossEntropyLoss always returns float32)

        return loss
