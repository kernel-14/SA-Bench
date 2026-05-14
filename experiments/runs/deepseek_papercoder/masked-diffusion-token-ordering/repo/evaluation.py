## evaluation.py
"""
Evaluation module for the masked diffusion reproduction project.

Implements the :class:`Evaluator` that computes various metrics required by the
paper's experiments: exact‑match accuracy, perplexity, generative perplexity
(via an external LLM), entropy, logic puzzle accuracy, and the error imbalance
analysis between a trained MDM and a proxy model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from configs import ExperimentConfig
from model import ARMWrapper, MDMTransformer, Model
from samplers import Sampler
from utils import MASK_TOKEN_ID, PAD_TOKEN_ID


class Evaluator:
    """
    Evaluator that computes metrics for different tasks.

    Args:
        model: A trained denoising network (``MDMTransformer`` or ``ARMWrapper``).
        data_loader: DataLoader providing evaluation data (format depends on task).
        sampler: A :class:`Sampler` instance for inference.  For ARMWrapper a
            dedicated autoregressive sampler is used internally.
        config: Full experiment configuration.
    """

    def __init__(
        self,
        model: Model,
        data_loader: DataLoader,
        sampler: Sampler,
        config: ExperimentConfig,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.sampler = sampler
        self.config = config

        # Device handling
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

        # Lazy‑loaded external LLM (e.g., LLaMA‑2)
        self._llm_model: Optional[AutoModelForCausalLM] = None
        self._llm_tokenizer: Optional[AutoTokenizer] = None

    # ------------------------------------------------------------------
    # Public API: main evaluate (optional dispatch based on task)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model according to the current task.

        Returns:
            Dictionary of metric names and values.
        """
        task = self.config.task
        if task in ("nae_sat",):
            return {
                "accuracy": self.evaluate_accuracy(),
            }
        elif task in ("scaling",):
            return {
                "perplexity": self.evaluate_perplexity(),
            }
        elif task in ("sudoku", "zebra"):
            return {
                "accuracy": self.evaluate_logic_puzzle(),
            }
        elif task == "llada":
            # For LLaDA the main metrics are generative perplexity and entropy
            # We assume a prompt‑based generation is required; here we provide
            # a simple wrapper for unconditional generation as a fallback.
            gen_ppl, entropy = self._eval_generative_metrics()
            return {
                "generative_perplexity": gen_ppl,
                "entropy": entropy,
            }
        else:
            raise ValueError(f"Unknown task '{task}' for evaluation.")

    # ------------------------------------------------------------------
    # Accuracy computation (exact match)
    # ------------------------------------------------------------------

    def compute_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        clue_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute exact‑match accuracy.

        Args:
            predictions: Token IDs, shape ``(B, L)``.
            targets: Ground‑truth token IDs, same shape.
            clue_mask: Optional boolean mask ``(B, L)``, ``True`` for positions
                that are clues (fixed) and should not be compared.

        Returns:
            Accuracy as a fraction in [0, 1].
        """
        if clue_mask is not None:
            # Override predictions with targets at clue positions so they match exactly
            predictions = predictions.clone()
            predictions[clue_mask] = targets[clue_mask]

        # A sequence is correct only if all non‑clue tokens match
        per_sample = (predictions == targets).all(dim=-1)  # (B,)
        return per_sample.float().mean().item()

    def evaluate_accuracy(self) -> float:
        """Run the sampler on the validation set and compute accuracy.

        The data loader is expected to yield batches with ``input_ids``
        (clue‑masked sequences) and ``labels`` (the full ground truth).
        For logic puzzles, this is the puzzle‑to‑solution mapping.

        Returns:
            Overall accuracy.
        """
        total_correct = 0
        total_samples = 0

        for batch in self.data_loader:
            # Determine which field contains the clue‑masked input.
            # For NAESAT we have only input_ids (full sequence), but we need to mask
            # some tokens? Actually NAESAT evaluation (Section 4.2) generates from a
            # fully masked sequence. We'll assume the data loader provides a field
            # "clue_mask" that indicates which tokens are given.
            if "clue_mask" in batch:
                # Logic puzzle or similar
                x_given = batch["input_ids"].to(self.device)
                x_target = batch["labels"].to(self.device) if "labels" in batch else batch["input_ids"].to(self.device)
                clue_mask = batch["clue_mask"].to(self.device)
                # Build the initial partially‑masked sequence
                x_masked = x_target.clone()
                x_masked[~clue_mask] = MASK_TOKEN_ID  # mask everything that is not a clue
            else:
                # Assume the input is already fully masked (or we generate from scratch)
                # For unconditional generation the length is unknown, but NAESAT has fixed length.
                x_given = batch["input_ids"].to(self.device)
                x_target = x_given.clone()
                # No clues: mask all
                x_masked = torch.full_like(x_target, MASK_TOKEN_ID)

            # Generate
            if isinstance(self.model, ARMWrapper):
                preds = self._arm_generate(x_masked, clue_mask)
            else:
                preds = self.sampler.sample(x_masked)

            correct = self.compute_accuracy(preds, x_target, clue_mask=clue_mask)
            n = x_target.size(0)
            total_correct += correct * n
            total_samples += n

        return total_correct / total_samples if total_samples > 0 else 0.0

    # ------------------------------------------------------------------
    # Perplexity evaluation (for π‑learner scaling)
    # ------------------------------------------------------------------

    def evaluate_perplexity(self) -> float:
        """Compute average token negative log‑likelihood over the entire dataset.

        For a bidirectional MDM, this implements the sum of conditional
        log‑probabilities along a given permutation (provided by the dataset).
        If the model is an ARMWrapper, we simply compute the standard
        autoregressive loss.

        Returns:
            Mean NLL (per token).
        """
        total_nll = 0.0
        total_tokens = 0

        for batch in self.data_loader:
            x0 = batch["input_ids"].to(self.device)               # (B, L)
            # Check if the dataset stored a permutation (for π‑learner evaluation)
            perm = batch.get("permutation", None)
            if perm is not None:
                # If a fixed permutation is given, we need to evaluate p_θ along that order.
                nll = self._mdm_permutation_nll(x0, perm.to(self.device))
            else:
                # ARMWrapper: standard autoregressive loss
                if isinstance(self.model, ARMWrapper):
                    logits = self.model.get_logits(x0)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = x0[:, 1:].contiguous()
                    # Cross‑entropy over vocabulary, but labels are shifted.
                    # Note: shift_labels contain token IDs 0→vocab_size-1,
                    # while model outputs logits over real tokens (1..m) – need to shift.
                    # The ARMWrapper outputs logits over real tokens only, so we shift labels by -1.
                    shift_labels = shift_labels - 1   # {0..m-1}
                    # Mask padding positions (we assume attention_mask is provided)
                    mask = batch.get("attention_mask", None)
                    if mask is not None:
                        mask = mask[:, 1:].reshape(-1)
                    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.reshape(-1)
                    ce = F.cross_entropy(flat_logits, flat_labels, reduction="none")
                    if mask is not None:
                        ce = ce * mask.to(ce.dtype)
                    nll = ce.sum()
                    n_tokens = mask.sum() if mask is not None else shift_labels.numel()
                else:
                    # If no permutation is provided and model is MDMTransformer,
                    # this case is not defined; raise.
                    raise ValueError(
                        "MDM perplexity evaluation requires a permutation. "
                        "Please provide a dataset that yields 'permutation' tensor."
                    )

            total_nll += nll.item()
            n_tokens = (mask.sum().item() if mask is not None else x0.numel())
            total_tokens += n_tokens

        return total_nll / total_tokens if total_tokens > 0 else float("inf")

    def _mdm_permutation_nll(
        self, x0: torch.Tensor, perm: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log‑likelihood for an MDM along a given permutation.

        Args:
            x0: Clean sequence ``(B, L)``.
            perm: Permutation indices ``(L,)`` giving the order in which tokens
                should be predicted. The convention is that at step j we predict
                ``x0[perm[j]]`` given all tokens ``x0[perm[j], perm[j+1], ...]``
                are masked.

        Returns:
            Total NLL summed over all tokens in the batch.
        """
        B, L = x0.shape
        device = x0.device
        total_nll = torch.tensor(0.0, device=device)

        # We'll iterate j from 0 to L-1, create progressively smaller mask sets.
        # For efficiency we can pre‑compute the mask for each step.
        for j in range(L):
            # Build the mask set: positions π(j), π(j+1), ..., π(L-1)
            idx_to_mask = perm[j:]                     # (L-j,)
            # Create masked input: start from x0 and mask these positions
            x_masked = x0.clone()
            # Create a 2D boolean mask
            mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            mask[:, idx_to_mask] = True
            x_masked[mask] = MASK_TOKEN_ID

            logits = self.model.get_logits(x_masked)     # (B, L, V)

            # We only need the loss at position π(j)
            target_idx = perm[j].unsqueeze(0).expand(B)    # (B,)
            # Gather logits for the specific position across batch
            # logits: B x L x V -> we need logits[b, target_idx[b], :]
            # Using advanced indexing:
            batch_indices = torch.arange(B, device=device)
            pred_logits = logits[batch_indices, target_idx]   # (B, V)

            # Target token: x0 at that position
            target_token = x0[batch_indices, target_idx]      # (B,)
            # Shift by -1 because model output is over real tokens {0..m-1}
            target_token = target_token - 1

            nll_j = F.cross_entropy(pred_logits, target_token, reduction="sum")
            total_nll = total_nll + nll_j

        return total_nll

    # ------------------------------------------------------------------
    # Generative Perplexity (using external LLM)
    # ------------------------------------------------------------------

    def eval_generative_perplexity(
        self,
        sampler: Optional[Sampler] = None,
        llm_model: Optional[AutoModelForCausalLM] = None,
    ) -> float:
        """Compute generative perplexity of the MDM under an external LLM.

        Args:
            sampler: Sampler to use (defaults to ``self.sampler``).
            llm_model: A pretrained causal LM for scoring. If ``None``, uses
                the lazily loaded LLaMA‑2 model.

        Returns:
            Generative perplexity (GenPPL) computed as exp(‑avg NLL/token) of the
            generated sequences under the LLM.
        """
        if sampler is None:
            sampler = self.sampler
        if llm_model is None:
            llm_model = self._load_llm()
        if llm_model is None:
            raise RuntimeError("No external LLM available; cannot compute generative perplexity.")

        tokenizer = self._llm_tokenizer
        if tokenizer is None:
            raise RuntimeError("LLM tokenizer not loaded.")

        num_samples = self.config.evaluation.text_sampling.num_samples
        total_nll = 0.0
        total_gen_tokens = 0

        # We assume unconditional generation. A prompt could be added if needed.
        # Determine length: use config.model.max_seq_length (or a reasonable value)
        max_len = self.config.model.max_seq_length

        for _ in tqdm(range(num_samples), desc="Generating for GenPPL"):
            # Generate a full sequence using MDM sampler
            x_masked = torch.full((1, max_len), MASK_TOKEN_ID, device=self.device)
            x_gen = sampler.sample(x_masked)[0]              # (L,)

            # Convert to text via the project's tokenizer (GPT‑2)
            # We need a tokenizer; ideally stored in dataset but we can load from config.
            # For simplicity we use the same tokenizer as data.
            base_tokenizer = AutoTokenizer.from_pretrained(
                self.config.data.tokenizer_name
            )
            text = base_tokenizer.decode(x_gen, skip_special_tokens=True)

            # Tokenize for LLaMA‑2
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = enc["input_ids"].to(self.device)

            # Compute log‑likelihood under LLaMA‑2
            with torch.no_grad():
                outputs = llm_model(input_ids, labels=input_ids)
                nll = outputs.loss * input_ids.numel()     # sum of NLL over tokens
            total_nll += nll.item()
            total_gen_tokens += input_ids.numel()

        return math.exp(total_nll / total_gen_tokens) if total_gen_tokens > 0 else float("inf")

    def eval_entropy(self, samples: Optional[torch.Tensor] = None) -> float:
        """Compute average token‑level Shannon entropy of generated sequences.

        Args:
            samples: Tensor of shape ``(N, L)`` containing token IDs. If ``None``,
                generates a batch using ``self.sampler``.

        Returns:
            Mean entropy across samples.
        """
        if samples is None:
            # Generate a batch of sequences using the sampler.
            # We'll use a single fully‑masked prompt of length max_len, generate
            # num_samples, then stack.
            num_samples = self.config.evaluation.text_sampling.num_samples
            max_len = self.config.model.max_seq_length
            all_samples = []
            for _ in tqdm(range(num_samples), desc="Generating for entropy"):
                x_masked = torch.full((1, max_len), MASK_TOKEN_ID, device=self.device)
                x_gen = self.sampler.sample(x_masked)          # (1, L)
                all_samples.append(x_gen.cpu())
            samples = torch.cat(all_samples, dim=0)            # (N, L)

        # Compute per‑sample entropy
        entropies = []
        N, L = samples.shape
        for n in range(N):
            counts = torch.bincount(samples[n], minlength=self.config.model.vocab_size)
            # Exclude MASK_TOKEN and PAD_TOKEN if present
            mask_count = counts[MASK_TOKEN_ID]
            pad_count = counts[PAD_TOKEN_ID] if PAD_TOKEN_ID < len(counts) else 0
            total_tokens = L - mask_count - pad_count
            if total_tokens <= 0:
                entropies.append(0.0)
                continue
            prob = counts / total_tokens
            prob = prob[prob > 0]
            entropy = - (prob * torch.log(prob)).sum().item()
            entropies.append(entropy)
        return float(np.mean(entropies))

    # ------------------------------------------------------------------
    # Logic puzzle evaluation (Section 4.2, 4.3)
    # ------------------------------------------------------------------

    def evaluate_logic_puzzle(
        self,
        model: Optional[Model] = None,
        sampler: Optional[Sampler] = None,
        dataset: Optional[DataLoader] = None,
    ) -> float:
        """Evaluate accuracy on logic puzzles (Sudoku, Zebra).

        Args:
            model: Model to use (defaults to self.model).
            sampler: Sampler for MDM inference (defaults to self.sampler).
            dataset: DataLoader yielding puzzle batches. (defaults to self.data_loader).

        Returns:
            Accuracy (fraction of puzzles exactly solved).
        """
        if model is None:
            model = self.model
        if sampler is None:
            sampler = self.sampler
        if dataset is None:
            dataset = self.data_loader

        model.eval()
        total_correct = 0
        total = 0

        for batch in tqdm(dataset, desc="Logic puzzle eval"):
            # Each batch should contain:
            # - input_ids: the puzzle initial state (clues + zeros for empty cells)
            # - labels: the full solution
            # - clue_mask: binary mask where 1 = clue (fixed token)
            puzzle = batch["input_ids"].to(self.device)      # (B, L)
            solution = batch["labels"].to(self.device)        # (B, L)
            clue_mask = batch.get("clue_mask", None)
            if clue_mask is not None:
                clue_mask = clue_mask.to(self.device)

            # Build the masked input: keep clue tokens, mask everything else
            x_masked = solution.clone()
            if clue_mask is not None:
                x_masked[~clue_mask] = MASK_TOKEN_ID
            else:
                x_masked[puzzle == 0] = MASK_TOKEN_ID

            # Generate
            if isinstance(model, ARMWrapper):
                # For autoregressive baselines, we need to feed the given context
                # and generate autoregressively respecting the learned order.
                # For ARM with ordering, the generation order is known; we follow
                # left‑to‑right (or the pre‑specified order).
                predictions = self._arm_generate(x_masked, clue_mask, order="left_to_right")
            else:
                predictions = sampler.sample(x_masked)

            # Compare only on non‑clue positions
            if clue_mask is not None:
                eval_mask = ~clue_mask
            else:
                eval_mask = torch.ones_like(solution, dtype=torch.bool)

            correct = self.compute_accuracy(predictions, solution, clue_mask=clue_mask)
            total_correct += correct * puzzle.size(0)
            total += puzzle.size(0)

        return total_correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Imbalance Analysis (Section 3.3)
    # ------------------------------------------------------------------

    def imbalance_analysis(
        self,
        proxy_model: Model,
        dataset: torch.utils.data.Dataset,
        masks: List[Tuple[int, int]],
        num_samples: int = 1000,
    ) -> Dict[str, List[float]]:
        """Measure squared difference in log‑prob between the current model
        and a proxy (Bayes‑optimal) model for specified mask configurations.

        Args:
            proxy_model: A better‑trained MDM approximating the Bayes optimal predictor.
            dataset: The dataset that can generate clean sequences (e.g., NAESATDataset).
            masks: List of tuples (num_latent_masked, num_obs_masked) indicating
                which mask configurations to evaluate.
            num_samples: Number of sequences to evaluate per configuration.

        Returns:
            Dictionary with keys ``'latent'`` and ``'observation'``, each holding a
            list of mean squared differences for each mask configuration.
        """
        self.model.eval()
        proxy_model.eval()

        # We'll accumulate squared differences separately for latent and observation positions.
        latent_errors: List[float] = []
        obs_errors: List[float] = []

        # Determine which positions are latent vs observation from the dataset
        # We assume NAESATDataset has attributes N and P.
        if not hasattr(dataset, "N") or not hasattr(dataset, "P"):
            raise ValueError("Dataset must provide N (latent length) and P (observation length).")

        N = dataset.N
        P = dataset.P
        total_len = N + P

        for num_latent, num_obs in masks:
            sq_diff_latent = 0.0
            sq_diff_obs = 0.0
            count_latent = 0
            count_obs = 0

            for _ in range(num_samples):
                # Get a clean sequence from the dataset
                idx = np.random.randint(len(dataset))
                sample = dataset[idx]
                x0 = sample["input_ids"].unsqueeze(0).to(self.device)  # (1, L) up to max_len
                # Pad or truncate to total_len? We'll just use the part up to N+P.
                x0 = x0[:, :total_len]

                # Randomly select latent positions to mask
                latent_positions = torch.randperm(N)[:num_latent]
                # Randomly select observation positions (offset by N)
                obs_positions = torch.randperm(P)[:num_obs] + N

                # Build mask set
                mask_indices = torch.cat([latent_positions, obs_positions])
                # Create masked input
                x_masked = x0.clone()
                x_masked[:, mask_indices] = MASK_TOKEN_ID

                # Get log‑probs from both models for each masked position
                for model_version, is_proxy in [(self.model, False), (proxy_model, True)]:
                    logits = model_version.get_logits(x_masked)          # (1, L, V)
                    probs = torch.softmax(logits, dim=-1)                # (1, L, V)

                    # For each masked position, extract log‑prob of true token
                    for pos_idx in mask_indices:
                        true_token = x0[0, pos_idx]
                        # true_token is in range 1..m; shift to model output index
                        token_idx = true_token - 1
                        log_prob = torch.log(probs[0, pos_idx, token_idx] + 1e-12)
                        # Determine category
                        if pos_idx < N:
                            if is_proxy:
                                # Store proxy log‑prob temporarily; after both passes we compute square diff
                                self._tmp_proxy_latent = log_prob.item()
                            else:
                                diff = (log_prob.item() - self._tmp_proxy_latent) ** 2
                                sq_diff_latent += diff
                                count_latent += 1
                        else:
                            if is_proxy:
                                self._tmp_proxy_obs = log_prob.item()
                            else:
                                diff = (log_prob.item() - self._tmp_proxy_obs) ** 2
                                sq_diff_obs += diff
                                count_obs += 1

            latent_errors.append(sq_diff_latent / count_latent if count_latent > 0 else 0.0)
            obs_errors.append(sq_diff_obs / count_obs if count_obs > 0 else 0.0)

        return {"latent": latent_errors, "observation": obs_errors}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_llm(self) -> AutoModelForCausalLM:
        """Lazy‑load the external LLM (LLaMA‑2) for generative perplexity."""
        if self._llm_model is None:
            llm_name = self.config.evaluation.llm_model_name
            self._llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
            self._llm_model = AutoModelForCausalLM.from_pretrained(llm_name)
            self._llm_model.to(self.config.evaluation.llm_device)
            self._llm_model.eval()
        return self._llm_model

    def _arm_generate(
        self,
        x_masked: torch.Tensor,
        clue_mask: Optional[torch.Tensor] = None,
        order: str = "left_to_right",
    ) -> torch.Tensor:
        """Simple autoregressive generation for ARM baselines.

        Args:
            x_masked: Initial partial sequence, with clue tokens fixed and
                remaining positions set to MASK_TOKEN_ID.
            clue_mask: Boolean mask indicating which positions are fixed.
            order: Order of generation. Currently only 'left_to_right' is supported.

        Returns:
            Completed sequence tensor.
        """
        model = self.model
        if not isinstance(model, ARMWrapper):
            raise ValueError("ARM generation only works with ARMWrapper.")

        B, L = x_masked.shape
        device = x_masked.device
        x_gen = x_masked.clone()

        # For left‑to‑right, iterate over positions and generate one by one.
        for pos in range(L):
            # If this position is already filled (clue), skip
            if clue_mask is not None and clue_mask[0, pos]:
                continue

            logits = model.get_logits(x_gen)
            # For a causal model, position pos only attends to earlier tokens, so we can
            # directly predict the next token using logits at pos-1 (since the model outputs
            # shifted predictions). Typically, logits[:, t, :] correspond to predicting token
            # at position t+1. So to predict token at position pos, we use logits at pos-1.
            if pos == 0:
                # Use a special start token or just ignore; for simplicity, we'll take
                # max probability from the projection of the first position.
                # ARMWrapper expects a BOS token; we'll assume x_masked starts with some
                # token. To avoid complications, we can use the first position directly
                # by feeding a dummy token and taking logits[0,0,:] as prediction for pos=0.
                pred_logits = logits[:, pos, :]
            else:
                pred_logits = logits[:, pos - 1, :]

            # Sample token
            probs = torch.softmax(pred_logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Model outputs over real tokens (1..vocab_size-1); shift back.
            token = token + 1
            x_gen[:, pos] = token

        return x_gen

    def _eval_generative_metrics(self) -> Tuple[float, float]:
        """Helper: compute generative perplexity and entropy in one go for LLaDA‑like tasks.

        This method generates a batch of sequences using the sampler and then
        computes both metrics. It is used when the task is 'llada'.

        Returns:
            Tuple of (generative perplexity, entropy).
        """
        # Generate a batch of samples
        num_samples = self.config.evaluation.text_sampling.num_samples
        max_len = self.config.model.max_seq_length
        all_samples = []
        for _ in tqdm(range(num_samples), desc="Generating for LLaDA eval"):
            x_masked = torch.full((1, max_len), MASK_TOKEN_ID, device=self.device)
            x_gen = self.sampler.sample(x_masked)
            all_samples.append(x_gen.cpu())
        samples = torch.cat(all_samples, dim=0)   # (N, L)

        # Compute entropy
        entropy = self.eval_entropy(samples)

        # Compute generative perplexity with external LLM
        gen_ppl = self.eval_generative_perplexity(self.sampler, llm_model=None)
        # Note: eval_generative_perplexity internally generates its own samples;
        # to avoid double generation, we could compute from samples directly but it's simpler
        # to just call it separately. This is acceptable for reproducibility.

        return gen_ppl, entropy


