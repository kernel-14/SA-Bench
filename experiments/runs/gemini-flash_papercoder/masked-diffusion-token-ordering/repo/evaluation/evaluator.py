import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Callable

# Placeholder for Config, TransformerMDM, PiLearnerARM, Inferrer, and NoiseScheduler
# to avoid circular imports. In main.py, the actual imports will be used.
# For this file's standalone integrity and type hinting, placeholders are used.
class _ConfigPlaceholder:
    def get(self, key: str, default: Any = None) -> Any:
        raise NotImplementedError("Placeholder: Use actual Config object.")
    @property
    def config_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Placeholder: Use actual Config object.")

class _TransformerMDMPlaceholder(nn.Module):
    def __init__(self, config: _ConfigPlaceholder) -> None: super().__init__()
    def forward(self, x_t: torch.Tensor, masked_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("Placeholder: Use actual TransformerMDM object.")

class _PiLearnerARMPlaceholder(nn.Module):
    def __init__(self, config: _ConfigPlaceholder) -> None: super().__init__()
    def forward(self, x_pi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Placeholder: Use actual PiLearnerARM object.")
    def compute_likelihood(self, x0: torch.Tensor, permutation: List[int]) -> torch.Tensor:
        raise NotImplementedError("Placeholder: Use actual PiLearnerARM object.")

class _InferrerPlaceholder:
    def __init__(self, config: _ConfigPlaceholder, model: _TransformerMDMPlaceholder, tokenizer: Any, noise_schedule: Any) -> None: pass
    def vanilla_inference(self, initial_x_t: torch.Tensor, num_steps: int) -> torch.Tensor:
        raise NotImplementedError("Placeholder: Use actual Inferrer object.")
    def adaptive_inference(self, initial_x_t: torch.Tensor, strategy: str, num_steps: int) -> torch.Tensor:
        raise NotImplementedError("Placeholder: Use actual Inferrer object.")

class _NoiseSchedulerPlaceholder:
    def get_alpha(self, t: float) -> float: raise NotImplementedError
    def get_mask_prob(self, t: float) -> float: raise NotImplementedError

# Re-assign for type hinting within this module.
# In the actual project, these would be:
# from config import Config
# from models.transformer_mdm import TransformerMDM
# from models.pi_learner_arm import PiLearnerARM
# from inference.inferrer import Inferrer
# from utils.noise_schedules import NoiseScheduler
Config = _ConfigPlaceholder
TransformerMDM = _TransformerMDMPlaceholder
PiLearnerARM = _PiLearnerARMPlaceholder
Inferrer = _InferrerPlaceholder
NoiseScheduler = _NoiseSchedulerPlaceholder

# Get logger instance. The logger is set up in utils/logger.py and retrieved here.
logger = logging.getLogger("MDM_Project_Logger")


class Evaluator:
    """
    The Evaluator class is responsible for assessing model performance across different
    tasks and inference strategies. It computes metrics like accuracy, perplexity,
    and entropy, and orchestrates the generation of samples (if applicable).
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        test_loader: DataLoader,
        inferrer: Optional[Inferrer] = None, # Inferrer is only needed for MDM evaluation
        tokenizer: Any = None,
        llama_model_for_ppl: Any = None,
        llama_tokenizer_for_ppl: Any = None # Added for consistency with llama_model_for_ppl
    ) -> None:
        """
        Initializes the Evaluator with necessary components for evaluation.

        Args:
            config (Config): The global configuration object.
            model (nn.Module): The model to be evaluated (TransformerMDM or PiLearnerARM).
            test_loader (DataLoader): DataLoader for the test dataset.
            inferrer (Optional[Inferrer]): An instance of the Inferrer class for MDM generation.
                                           Required for MDM evaluations.
            tokenizer (Any, optional): A tokenizer instance for text processing.
            llama_model_for_ppl (Any, optional): An external language model (e.g., LLaMA2-7B)
                                                 for computing generative perplexity.
            llama_tokenizer_for_ppl (Any, optional): The tokenizer corresponding to `llama_model_for_ppl`.
        """
        self.config: Config = config
        self.model: nn.Module = model
        self.test_loader: DataLoader = test_loader
        self.inferrer: Optional[Inferrer] = inferrer
        self.tokenizer: Any = tokenizer
        self.llama_model_for_ppl: Any = llama_model_for_ppl
        self.llama_tokenizer_for_ppl: Any = llama_tokenizer_for_ppl

        self.device: str = self.config.get('general.device', 'cpu')
        self.model_type: str = self.config.get('model.model_type')
        self.dataset_type: str = self.config.get('data.dataset_type', 'unknown')
        self.max_sequence_length: int = self.config.get('data.max_sequence_length', 1)
        self.vocab_size: int = self.config.get('data.vocab_size', 1)
        self.mask_token_id: int = self.config.get('data.mask_token_id', 0)

        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)

        if self.llama_model_for_ppl:
            self.llama_model_for_ppl.eval()
            self.llama_model_for_ppl.to(self.device)
            logger.info(f"LLaMA model for perplexity loaded on device: {self.device}")
        elif "perplexity" in self.config.get('evaluation.metrics', []):
            logger.warning("Perplexity evaluation requested, but no LLaMA model provided. "
                           "Please provide 'llama_model_for_ppl' and 'llama_tokenizer_for_ppl' "
                           "or ensure 'config.evaluation.perplexity_llm_model_path' is set and valid.")

        logger.info(f"Evaluator initialized for model type: {self.model_type}, dataset type: {self.dataset_type}")

    def evaluate_accuracy(self, generated_outputs: List[torch.Tensor], true_labels: List[torch.Tensor]) -> float:
        """
        Calculates the accuracy of generated_outputs against true_labels based on the task type.

        Args:
            generated_outputs (List[torch.Tensor]): A list of torch.Tensors, where each tensor
                                                    represents a generated sequence.
            true_labels (List[torch.Tensor]): A list of torch.Tensors, where each tensor
                                               represents the ground truth.

        Returns:
            float: The calculated accuracy (percentage).
        """
        if not generated_outputs or not true_labels:
            logger.warning("No generated outputs or true labels provided for accuracy evaluation. Returning 0.0.")
            return 0.0
        
        total_samples: int = len(generated_outputs)
        correct_predictions: int = 0
        
        # Ensure all tensors are on CPU for consistent comparison if device differs
        generated_outputs_cpu = [out.cpu() for out in generated_outputs]
        true_labels_cpu = [label.cpu() for label in true_labels]

        if self.dataset_type == "lo_naesat":
            N: int = self.config.get('data.lo_naesat.N', 0)
            if N == 0:
                logger.warning("L&O-NAESAT 'N' (latent tokens) is 0 in config, cannot distinguish observation tokens. Calculating full sequence match accuracy.")
                for gen, true in zip(generated_outputs_cpu, true_labels_cpu):
                    if torch.equal(gen, true):
                        correct_predictions += 1
                return (correct_predictions / total_samples) * 100.0 if total_samples > 0 else 0.0

            total_observation_tokens: int = 0
            correct_observation_tokens: int = 0
            
            for gen, true in zip(generated_outputs_cpu, true_labels_cpu):
                # Ensure sequences are long enough
                if gen.size(-1) < N or true.size(-1) < N:
                    logger.warning(f"L&O-NAESAT sequence length ({gen.size(-1)}) too short for N={N}. Skipping sample.")
                    continue
                
                # Compare only observation tokens (from N to end)
                gen_obs = gen[N:]
                true_obs = true[N:]
                
                total_observation_tokens += gen_obs.numel()
                correct_observation_tokens += (gen_obs == true_obs).sum().item()

            return (correct_observation_tokens / total_observation_tokens) * 100.0 if total_observation_tokens > 0 else 0.0

        elif self.dataset_type in ["sudoku", "zebra"]:
            # A puzzle is considered correctly solved if all tokens match the ground truth
            for gen, true in zip(generated_outputs_cpu, true_labels_cpu):
                if torch.equal(gen, true):
                    correct_predictions += 1
            return (correct_predictions / total_samples) * 100.0 if total_samples > 0 else 0.0

        elif self.dataset_type == "llada_tasks":
            # For LLaDA tasks, a simple exact sequence match is used for now.
            # More complex task-specific evaluations (e.g., HumanEval pass@k) would
            # require external scripts or parsing of string outputs.
            for gen, true in zip(generated_outputs_cpu, true_labels_cpu):
                # Optionally, convert to strings for more robust comparison, handling potential padding differences.
                # Assuming `generated_outputs` and `true_labels` are token IDs and represent the full output.
                if torch.equal(gen, true):
                    correct_predictions += 1
            return (correct_predictions / total_samples) * 100.0 if total_samples > 0 else 0.0

        else:
            logger.warning(f"Accuracy evaluation not implemented for dataset type: {self.dataset_type}. Calculating full sequence match accuracy.")
            for gen, true in zip(generated_outputs_cpu, true_labels_cpu):
                if torch.equal(gen, true):
                    correct_predictions += 1
            return (correct_predictions / total_samples) * 100.0 if total_samples > 0 else 0.0


    def evaluate_perplexity(self, generated_samples: List[torch.Tensor]) -> float:
        """
        Computes the generative perplexity (GenPPL) of generated text samples
        using an external language model (LLaMA2-7B or configured).

        Args:
            generated_samples (List[torch.Tensor]): A list of torch.Tensors, where each
                                                    tensor represents a tokenized generated text sequence.

        Returns:
            float: The calculated perplexity. Returns 0.0 if LLaMA model not available.
        """
        if not self.llama_model_for_ppl or not self.llama_tokenizer_for_ppl:
            logger.warning("LLaMA model/tokenizer for perplexity not initialized. Cannot compute perplexity. Returning 0.0.")
            return 0.0

        if not generated_samples:
            logger.warning("No generated samples provided for perplexity evaluation. Returning 0.0.")
            return 0.0

        total_nll: float = 0.0
        total_tokens: int = 0
        
        # Ensure LLaMA model is in eval mode and on device
        self.llama_model_for_ppl.eval()
        self.llama_model_for_ppl.to(self.device)

        with torch.no_grad():
            for sample_tokens in generated_samples:
                # Add batch dimension and move to device
                input_ids = sample_tokens.unsqueeze(0).to(self.device) # (1, seq_len)
                
                # Causal LM expects targets shifted by one
                labels = input_ids.clone()
                labels[labels == self.llama_tokenizer_for_ppl.pad_token_id] = -100 # Ignore padding tokens for loss

                # Get model outputs (logits)
                outputs = self.llama_model_for_ppl(input_ids, labels=labels)
                
                # outputs.loss is already the mean negative log-likelihood for non-masked tokens
                # We need to compute sum of NLL to then average over all tokens in all samples
                
                # The loss calculated by the model is usually the average over non-masked tokens.
                # To get the sum, we multiply by the number of non-masked tokens.
                num_non_pad_tokens = (input_ids != self.llama_tokenizer_for_ppl.pad_token_id).sum().item()
                if num_non_pad_tokens > 0:
                    total_nll += outputs.loss.item() * num_non_pad_tokens
                    total_tokens += num_non_pad_tokens
                else:
                    logger.warning(f"Generated sample had no non-padding tokens. Skipping NLL calculation for this sample.")

        if total_tokens == 0:
            logger.warning("No valid tokens to compute perplexity over. Returning 0.0.")
            return 0.0

        average_nll: float = total_nll / total_tokens
        perplexity: float = math.exp(average_nll)
        
        return perplexity

    def evaluate_entropy(self, generated_samples: List[torch.Tensor]) -> float:
        """
        Measures the diversity of generated samples by computing their empirical entropy.
        Entropy = - sum(p_i * log(p_i)) where p_i is the probability of token i.

        Args:
            generated_samples (List[torch.Tensor]): A list of torch.Tensors, where each
                                                    tensor represents a tokenized generated text sequence.

        Returns:
            float: The average empirical entropy across all generated samples.
        """
        if not generated_samples:
            logger.warning("No generated samples provided for entropy evaluation. Returning 0.0.")
            return 0.0

        total_entropy: float = 0.0
        valid_samples_count: int = 0

        for sample_tokens in generated_samples:
            # Flatten the tensor and remove mask tokens if present, or padding
            clean_tokens = sample_tokens[sample_tokens != self.mask_token_id].cpu()
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                clean_tokens = clean_tokens[clean_tokens != self.tokenizer.pad_token_id]
            
            sequence_length: int = clean_tokens.numel()

            if sequence_length == 0:
                logger.debug("Skipping entropy calculation for an empty or fully masked/padded sequence.")
                continue

            # Count occurrences of each unique token
            unique_tokens, counts = clean_tokens.unique(return_counts=True)
            
            # Calculate probabilities
            probabilities: torch.Tensor = counts.float() / sequence_length
            
            # Compute entropy: -sum(p * log(p))
            # Use natural logarithm (log)
            entropy_value: float = (-probabilities * torch.log(probabilities)).sum().item()
            
            total_entropy += entropy_value
            valid_samples_count += 1

        return (total_entropy / valid_samples_count) if valid_samples_count > 0 else 0.0

    def _apply_custom_masking_for_error_imbalance(
        self,
        x0_batch: torch.Tensor,
        l_val: int, # number of latent tokens to mask
        N: int, # total latent tokens
        P: int # total observation tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a specific masking pattern for L&O-NAESAT error imbalance evaluation:
        randomly masks l latent tokens and l * (P/N) observation tokens.

        Args:
            x0_batch (torch.Tensor): The original (clean) input sequence batch. Shape: (batch_size, sequence_length).
            l_val (int): The number of latent tokens to mask.
            N (int): Total number of latent tokens in the sequence.
            P (int): Total number of observation tokens in the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_t (torch.Tensor): The partially masked sequence. Shape: (batch_size, sequence_length).
                - masked_mask_bool_tensor (torch.Tensor): A boolean tensor indicating
                                                        which positions were masked. Shape: (batch_size, sequence_length).
        """
        batch_size, sequence_length = x0_batch.size()
        x_t = x0_batch.clone()
        masked_mask_bool_tensor = torch.zeros_like(x0_batch, dtype=torch.bool)

        num_obs_to_mask = min(int(l_val * (P / N)), P) # Ensure not to mask more than available observation tokens

        for i in range(batch_size):
            # Select 'l_val' random latent positions to mask
            latent_indices = list(range(N)) # Latent tokens are at positions 0 to N-1
            selected_latent_indices = random.sample(latent_indices, l_val)
            
            # Select 'num_obs_to_mask' random observation positions to mask
            observation_indices = list(range(N, N + P)) # Observation tokens are at positions N to N+P-1
            selected_observation_indices = random.sample(observation_indices, num_obs_to_mask)
            
            # Combine all indices to mask for the current sample
            indices_to_mask = selected_latent_indices + selected_observation_indices
            
            # Apply mask
            x_t[i, indices_to_mask] = self.mask_token_id
            masked_mask_bool_tensor[i, indices_to_mask] = True

        return x_t, masked_mask_bool_tensor.to(self.device)


    def _evaluate_lo_naesat_error_imbalance(self, mdm_model: TransformerMDM, bayes_optimal_proxy_model: TransformerMDM) -> Tuple[float, float]:
        """
        Calculates the error imbalance metric for L&O-NAE-SAT distributions (Section 3.3).

        Args:
            mdm_model (TransformerMDM): The TransformerMDM trained normally.
            bayes_optimal_proxy_model (TransformerMDM): Another TransformerMDM trained
                                                        for much longer, acting as a proxy for p_data.

        Returns:
            Tuple[float, float]: Average error for latent positions, average error for observation positions.
        """
        N: int = self.config.get('data.lo_naesat.N')
        P: int = self.config.get('data.lo_naesat.P')
        l_val: int = self.config.get('evaluation.lo_naesat.l_val', 11) # Paper uses l=11 (Section C.2.1)
        num_eval_repeats: int = self.config.get('evaluation.lo_naesat.num_eval_repeats', 1000)

        total_latent_error: float = 0.0
        total_observation_error: float = 0.0
        count_latent_positions: int = 0
        count_observation_positions: int = 0

        mdm_model.eval()
        bayes_optimal_proxy_model.eval()

        with torch.no_grad():
            for _ in range(num_eval_repeats): # Repeat the process many times
                # Get a batch from the test loader
                try:
                    batch = next(iter(self.test_loader))
                except StopIteration:
                    logger.warning("L&O-NAESAT test_loader exhausted during error imbalance evaluation. Re-iterating.")
                    self.test_loader.dataset.prepare_data() # Re-generate data for LO_NAESAT if it's dynamic
                    batch = next(iter(self.test_loader))

                x0_batch: torch.Tensor = batch['input_ids'].to(self.device)
                
                # Apply custom masking for error imbalance
                x_t, masked_mask_bool_tensor = self._apply_custom_masking_for_error_imbalance(x0_batch, l_val, N, P)
                
                # Get predictions from both models
                mdm_logits: torch.Tensor = mdm_model(x_t)
                bayes_logits: torch.Tensor = bayes_optimal_proxy_model(x_t)

                # Get log probabilities for masked positions only
                mdm_log_probs = F.log_softmax(mdm_logits, dim=-1)
                bayes_log_probs = F.log_softmax(bayes_logits, dim=-1)

                for batch_idx in range(x0_batch.size(0)):
                    masked_indices_for_sample = torch.nonzero(masked_mask_bool_tensor[batch_idx]).squeeze(1)
                    
                    if masked_indices_for_sample.numel() == 0:
                        continue

                    # Extract log-probs for masked positions from both models
                    mdm_lp_masked = mdm_log_probs[batch_idx, masked_indices_for_sample, :]
                    bayes_lp_masked = bayes_log_probs[batch_idx, masked_indices_for_sample, :]

                    # Calculate squared L2 distance for each masked position
                    squared_errors = torch.norm(mdm_lp_masked - bayes_lp_masked, p=2, dim=-1)**2

                    # Accumulate errors based on whether they are latent or observation positions
                    for i, error in zip(masked_indices_for_sample.tolist(), squared_errors.tolist()):
                        if i < N: # Latent position
                            total_latent_error += error
                            count_latent_positions += 1
                        else: # Observation position
                            total_observation_error += error
                            count_observation_positions += 1
            
        avg_latent_error: float = (total_latent_error / count_latent_positions) if count_latent_positions > 0 else 0.0
        avg_observation_error: float = (total_observation_error / count_observation_positions) if count_observation_positions > 0 else 0.0

        return avg_latent_error, avg_observation_error


    def _arm_generate(self, initial_x: torch.Tensor, num_tokens_to_generate: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generates tokens autoregressively using the PiLearnerARM model in a greedy or sampled manner.
        This is a simplified generation for ARM puzzle solving (Sudoku/Zebra).

        Args:
            initial_x (torch.Tensor): The initial input sequence (e.g., partially filled puzzle).
                                      Shape: (1, sequence_length).
            num_tokens_to_generate (int): The number of tokens to generate to complete the sequence.
                                          In puzzles, this would be the number of masked tokens.
            temperature (float): Sampling temperature. Higher values lead to more randomness.

        Returns:
            torch.Tensor: The completed generated sequence. Shape: (1, sequence_length).
        """
        if not isinstance(self.model, PiLearnerARM):
            raise TypeError("This generation method is only for PiLearnerARM.")
        
        self.model.eval()
        generated_sequence = initial_x.clone().to(self.device) # (1, seq_len)
        
        # Identify positions that are still masked (target for generation)
        masked_positions = (generated_sequence[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
        
        if masked_positions.numel() == 0:
            logger.debug("ARM generation: No masked tokens in initial sequence. Returning as is.")
            return generated_sequence

        # Generate tokens one by one for masked positions.
        # This simple generation strategy iterates through masked positions sequentially.
        # A more sophisticated ARM generation might reorder masked positions dynamically or sample.
        for idx in masked_positions:
            with torch.no_grad():
                # The ARM model's forward expects a permuted sequence, but for standard greedy
                # generation, we just feed the current (partially filled) sequence.
                # Assuming the internal attention mechanism handles causal masking correctly up to `idx`.
                # For this simple generation, we are not applying a custom permutation,
                # so the model predicts based on the existing (partially filled) prefix.
                
                # A proper ARM `generate` would predict the *next* token in a causal sequence.
                # Here, we get logits for the *entire* sequence, and use the one at the current `idx`.
                
                # If `PiLearnerARM` was trained only on fully left-to-right, this generation would be LTR.
                # If it was trained on random permutations (as in `compute_likelihood`), this will be less direct.
                # For puzzle solving, the ARM (w/ ordering) needs to know the order of generation.
                # Let's assume a simple left-to-right fill for empty spots.

                # Generate one token at a time at the *next* available masked position
                # This treats `generated_sequence` as the prefix.
                # `self.model(generated_sequence)` will return logits for `(B, T, V)`.
                # We want the logits at the current `idx` to predict `idx+1` (if this were simple LTR)
                # or the `idx`-th position itself if the ARM is designed to predict *any* masked token
                # given a context. The paper's ARM is a standard causal LM.

                # Let's assume standard LTR generation: fill the first mask found
                # For simplicity, we assume we find the next masked position `idx`.
                
                output_logits = self.model(generated_sequence) # (1, seq_len, vocab_size)
                
                # Get logits for the current position `idx` where we want to generate a token
                logits_at_current_pos = output_logits[0, idx, :] # (vocab_size,)

                if temperature == 0.0:
                    next_token_id = torch.argmax(logits_at_current_pos, dim=-1).item()
                else:
                    probabilities = F.softmax(logits_at_current_pos / temperature, dim=-1)
                    next_token_id = torch.multinomial(probabilities, num_samples=1).item()
                
                generated_sequence[0, idx] = next_token_id
        
        return generated_sequence


    def run_all_evaluations(self, inference_strategies: List[str]) -> Dict[str, Any]:
        """
        Orchestrates the entire evaluation process based on the configured model type and tasks.

        Args:
            inference_strategies (List[str]): A list of strings indicating which MDM inference
                                             strategies to evaluate (e.g., "vanilla", "top_probability").

        Returns:
            Dict[str, Any]: A dictionary containing all evaluation outcomes.
        """
        results: Dict[str, Any] = {}
        self.model.eval() # Ensure model is in evaluation mode

        logger.info(f"Running evaluations for model type: {self.model_type}")

        with torch.no_grad():
            if isinstance(self.model, TransformerMDM):
                if self.inferrer is None:
                    raise ValueError("Inferrer must be provided for TransformerMDM evaluation.")

                num_inference_steps: int = self.config.get('inference.num_inference_steps', 50)
                
                # MDM Evaluation Loop
                for strategy in inference_strategies:
                    logger.info(f"Evaluating MDM with inference strategy: '{strategy}'")
                    all_generated_samples: List[torch.Tensor] = []
                    all_true_labels: List[torch.Tensor] = []

                    for batch_idx, batch in enumerate(self.test_loader):
                        x0_batch: torch.Tensor = batch['input_ids'].to(self.device) # Ground truth
                        true_labels_batch: torch.Tensor = batch['labels'].to(self.device) if 'labels' in batch else x0_batch
                        
                        # Create initial fully masked sequence (for single sample generation)
                        # We generate one by one for cleaner logic, then batch results
                        for i in range(x0_batch.size(0)):
                            initial_x_t = torch.full((1, self.max_sequence_length), self.mask_token_id, dtype=torch.long, device=self.device)
                            # Partially fill initial_x_t with known tokens for puzzle infilling scenarios
                            known_mask = (x0_batch[i] != self.mask_token_id)
                            initial_x_t[0, known_mask] = x0_batch[i, known_mask]


                            generated_sample: torch.Tensor
                            if strategy == "vanilla":
                                generated_sample = self.inferrer.vanilla_inference(initial_x_t, num_inference_steps)
                            else: # Adaptive strategies
                                generated_sample = self.inferrer.adaptive_inference(initial_x_t, strategy, num_inference_steps)
                            
                            all_generated_samples.append(generated_sample.squeeze(0)) # Remove batch dim
                            all_true_labels.append(true_labels_batch[i])

                    strategy_results: Dict[str, Any] = {}
                    if "accuracy" in self.config.get('evaluation.metrics', []):
                        accuracy = self.evaluate_accuracy(all_generated_samples, all_true_labels)
                        logger.info(f"  {strategy} Accuracy: {accuracy:.2f}%")
                        strategy_results['accuracy'] = accuracy
                    
                    if "perplexity" in self.config.get('evaluation.metrics', []) and self.dataset_type == "slimpajama":
                        perplexity = self.evaluate_perplexity(all_generated_samples)
                        logger.info(f"  {strategy} Perplexity: {perplexity:.2f}")
                        strategy_results['perplexity'] = perplexity
                    
                    if "entropy" in self.config.get('evaluation.metrics', []) and self.dataset_type == "slimpajama":
                        entropy = self.evaluate_entropy(all_generated_samples)
                        logger.info(f"  {strategy} Entropy: {entropy:.2f}")
                        strategy_results['entropy'] = entropy
                    
                    results[strategy] = strategy_results
                    self.logger.wandb.log({f"eval/{strategy}/{metric}": value for metric, value in strategy_results.items()})

                # Special Case: L&O-NAE-SAT Error Imbalance (Section 3.3)
                if self.dataset_type == "lo_naesat" and self.config.get('general.evaluate_error_imbalance', False):
                    logger.info("Evaluating L&O-NAESAT error imbalance...")
                    # Load Bayes optimal proxy model
                    bayes_proxy_model_path: Optional[str] = self.config.get('evaluation.lo_naesat.bayes_proxy_model_path')
                    bayes_optimal_proxy_model: Optional[TransformerMDM] = None
                    if bayes_proxy_model_path:
                        try:
                            # Re-initialize the model with the same config as the main model
                            bayes_optimal_proxy_model = TransformerMDM(self.config)
                            checkpoint = torch.load(bayes_proxy_model_path, map_location=self.device)
                            bayes_optimal_proxy_model.load_state_dict(checkpoint['model_state_dict'])
                            bayes_optimal_proxy_model.to(self.device).eval()
                            logger.info(f"Loaded Bayes optimal proxy model from {bayes_proxy_model_path}")
                        except Exception as e:
                            logger.error(f"Failed to load Bayes optimal proxy model: {e}. Skipping error imbalance evaluation.")
                            bayes_optimal_proxy_model = None
                    
                    if bayes_optimal_proxy_model:
                        latent_error, obs_error = self._evaluate_lo_naesat_error_imbalance(self.model, bayes_optimal_proxy_model)
                        logger.info(f"  L&O-NAESAT Error Imbalance: Latent Error = {latent_error:.4f}, Observation Error = {obs_error:.4f}")
                        results['lo_naesat_error_imbalance'] = {'latent_error': latent_error, 'observation_error': obs_error}
                        self.logger.wandb.log({
                            "eval/lo_naesat_error_imbalance/latent_error": latent_error,
                            "eval/lo_naesat_error_imbalance/observation_error": obs_error
                        })
            
            elif isinstance(self.model, PiLearnerARM):
                # ARM Evaluation Loop
                arm_results: Dict[str, Any] = {}
                if "perplexity" in self.config.get('evaluation.metrics', []) and self.dataset_type == "slimpajama":
                    logger.info("Evaluating ARM (Pi-learner) perplexity (using negative log-likelihood).")
                    total_log_likelihood: float = 0.0
                    total_tokens_for_ppl: int = 0
                    
                    # For PiLearner, we need to iterate over test_loader and compute likelihoods
                    for batch_idx, batch in enumerate(self.test_loader):
                        x0_batch: torch.Tensor = batch['input_ids'].to(self.device)
                        permutation_batch: List[List[int]] = batch.get('permutation', [list(range(self.max_sequence_length))] * x0_batch.size(0))
                        
                        # Flatten list of lists for compute_likelihood
                        flat_permutations: List[int] = []
                        for perm in permutation_batch:
                            flat_permutations.extend(perm)

                        if isinstance(permutation_batch, torch.Tensor): # if it's already a tensor from dataset
                             permutation_batch = permutation_batch.tolist()

                        # Compute likelihood per sequence
                        log_likelihoods_batch = self.model.compute_likelihood(x0_batch, permutation_batch[0] if isinstance(permutation_batch[0], list) else permutation_batch) # Pass 1D permutation for now for simplicity if all are same

                        total_log_likelihood += log_likelihoods_batch.sum().item()
                        total_tokens_for_ppl += x0_batch.size(0) * (x0_batch.size(1) - 1) # Each token (except first) contributes to likelihood

                    average_log_likelihood = total_log_likelihood / total_tokens_for_ppl
                    perplexity = math.exp(-average_log_likelihood) # PPL = exp(-average NLL)
                    logger.info(f"  ARM (Pi-learner) Perplexity: {perplexity:.2f}")
                    arm_results['perplexity'] = perplexity
                    self.logger.wandb.log({"eval/arm/perplexity": perplexity})

                if self.dataset_type in ["sudoku", "zebra"]:
                    logger.info("Evaluating ARM (Pi-learner) for puzzle solving accuracy.")
                    all_generated_samples: List[torch.Tensor] = []
                    all_true_labels: List[torch.Tensor] = []
                    
                    for batch_idx, batch in enumerate(self.test_loader):
                        x0_batch: torch.Tensor = batch['input_ids'].to(self.device) # Partially filled puzzle as initial_x
                        true_labels_batch: torch.Tensor = batch['labels'].to(self.device)
                        
                        for i in range(x0_batch.size(0)):
                            initial_puzzle_state = x0_batch[i].unsqueeze(0) # (1, seq_len)
                            # Number of tokens to generate for completion
                            num_masked = (initial_puzzle_state == self.mask_token_id).sum().item()
                            
                            generated_puzzle = self._arm_generate(initial_puzzle_state, num_masked, temperature=0.0) # Greedy generation
                            all_generated_samples.append(generated_puzzle.squeeze(0))
                            all_true_labels.append(true_labels_batch[i])

                    accuracy = self.evaluate_accuracy(all_generated_samples, all_true_labels)
                    logger.info(f"  ARM (Pi-learner) Accuracy: {accuracy:.2f}%")
                    arm_results['accuracy'] = accuracy
                    self.logger.wandb.log({"eval/arm/accuracy": accuracy})

                results['arm'] = arm_results

            else:
                logger.warning(f"Unsupported model type for evaluation: {type(self.model)}. No evaluations performed.")

        logger.info("All evaluations complete.")
        return results

