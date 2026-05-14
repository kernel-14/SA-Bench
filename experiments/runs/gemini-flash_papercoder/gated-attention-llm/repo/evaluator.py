import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import Optional, List, Dict, Any, Tuple
import json

# Local imports
try:
    from config import Config
    from model.gated_transformer import GatedTransformer
    from data_loader import DataLoader
    # utils is not directly used in Evaluator but can be imported if needed for helper functions
    # from utils import calculate_model_parameters
except ImportError as e:
    print(f"Failed to import local modules: {e}")
    print("Ensure config.py, model/gated_transformer.py, data_loader.py are accessible.")
    # Dummy classes for standalone testing/IDE syntax checking.
    # In a real run, these must be properly imported.
    class Config:
        def __init__(self):
            self.model = type('ModelConfig', (object,), {
                'vocab_size': 32000,
                'max_seq_len': 4096,
            })()
            self.evaluation = type('EvaluationConfig', (object,), {
                'benchmarks': [],
                'ruler_benchmark_enabled': False,
                'attention_sink_analysis_enabled': False,
                'massive_activation_analysis_enabled': False,
                'gating_score_analysis_enabled': False,
            })()
            self.training = type('TrainingConfig', (object,), {
                'global_batch_size': 1,
                'mixed_precision': 'no',
            })()
            self.gating_enabled = True # For get_gating_metrics dummy
            self.output_dir = "./dummy_output" # For lm_eval output


    class GatedTransformer(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.lm_head = nn.Linear(config.model.vocab_size, config.model.vocab_size)
            self.num_layers = 1 # Dummy for analysis loop
            self.config = config # Dummy config to access evaluation flags

        def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            dummy_logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 32000, device=input_ids.device)
            dummy_loss = None
            if labels is not None:
                # Simulate mean loss over active tokens
                num_active_tokens = (labels != -100).sum()
                if num_active_tokens > 0:
                    dummy_loss = torch.randn(1, device=input_ids.device) * (input_ids.shape[0] * input_ids.shape[1]) / num_active_tokens # Scale to represent mean per active token
                else:
                    dummy_loss = torch.tensor(0.0, device=input_ids.device)
            return dummy_logits, dummy_loss, None

        def get_gating_metrics(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
            # Dummy implementation for get_gating_metrics
            metrics = {
                "gating_scores": [],
                "attention_weights": [],
                "massive_activations_ffn": [],
                "massive_activations_attn": [],
            }
            # Populate with dummy data for one layer
            dummy_batch_size = input_ids.shape[0]
            dummy_seq_len = input_ids.shape[1]
            dummy_d_model = 2048 # A common d_model for 1.7B
            dummy_q_heads = 32
            dummy_head_dim = 128

            if self.config.gating_enabled and self.config.evaluation.gating_score_analysis_enabled:
                metrics["gating_scores"].append(torch.rand(dummy_batch_size, dummy_seq_len, dummy_q_heads, dummy_head_dim).cpu())
            if self.config.evaluation.attention_sink_analysis_enabled:
                metrics["attention_weights"].append(torch.rand(dummy_batch_size, dummy_q_heads, dummy_seq_len, dummy_seq_len).cpu())
            if self.config.evaluation.massive_activation_analysis_enabled:
                metrics["massive_activations_ffn"].append(torch.rand(dummy_batch_size, dummy_seq_len, dummy_d_model).cpu())
                metrics["massive_activations_attn"].append(torch.rand(dummy_batch_size, dummy_seq_len, dummy_d_model).cpu())
            return metrics


    class DataLoader:
        def __init__(self, config: Config):
            self.config = config
            self.tokenizer = type('Tokenizer', (object,), {'pad_token_id': 0, 'eos_token_id': 1})() # Dummy tokenizer
            self.max_seq_len = config.model.max_seq_len
            self.global_batch_size = config.training.global_batch_size

        def get_eval_dataloader(self) -> torch.utils.data.DataLoader:
            input_ids = torch.randint(0, 32000, (self.global_batch_size, self.max_seq_len))
            attention_mask = torch.ones((self.global_batch_size, self.max_seq_len))
            labels = torch.randint(0, 32000, (self.global_batch_size, self.max_seq_len))
            # Set some labels to -100 for ignore_index
            labels[:, 0] = -100 # Example: ignore first token
            labels[0, -10:] = -100 # Example: ignore some last tokens
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(input_ids, attention_mask, labels),
                batch_size=self.global_batch_size,
                shuffle=False
            )


# Conditional import for lm_eval
_LM_EVAL_AVAILABLE = False
try:
    import lm_eval.tasks
    import lm_eval.models.huggingface
    from lm_eval import evaluator
    from lm_eval.base import LM
    _LM_EVAL_AVAILABLE = True
except ImportError:
    print("Warning: 'lm_eval' library not found. Benchmark evaluation will be disabled.")
    # Define LM as object if lm_eval is not available to avoid NameError
    # This ensures the CustomLMEvalModel definition doesn't break if lm_eval.base.LM is missing
    class LM:
        def __init__(self): pass
        @property
        def eot_token_id(self): return None
        @property
        def max_length(self): return None
        @property
        def max_gen_toks(self): return None
        @property
        def batch_size_per_gpu(self): return None
        def tok_encode(self, string: str, **kwargs) -> List[int]: return []
        def tok_decode(self, tokens: List[int], **kwargs) -> str: return ""
        def _model_call(self, inps: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): return None
        def _model_generate(self, inps, gen_kwargs): return None


if _LM_EVAL_AVAILABLE:
    class CustomLMEvalModel(LM):
        """
        A wrapper around GatedTransformer to make it compatible with lm_eval.
        Implements the LM abstract base class from lm_eval.
        """
        def __init__(self, model: GatedTransformer, tokenizer, batch_size: int, device: torch.device):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.device = device
            self.model.eval() # Ensure model is in eval mode

            # A GatedTransformer does not inherently have a `generate` method.
            # To support generation tasks, a custom `generate` logic would be needed.
            # For now, if _model_generate is called, it might raise an error or return dummy.
            # For PPL tasks, _model_call (for logits) is sufficient.
            if not hasattr(self.model, 'generate'):
                def dummy_generate(*args, **kwargs):
                    raise NotImplementedError("GatedTransformer does not have a 'generate' method. "
                                              "Generation tasks with lm_eval are not supported.")
                self.model.generate = dummy_generate


        @property
        def eot_token_id(self):
            # End of Text token ID (often the EOS token)
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            return self.model.config.model.max_seq_len

        @property
        def max_gen_toks(self):
            # Max tokens to generate per call
            return 256 # Default for lm_eval, can be configured in lm_eval harness calls

        @property
        def batch_size_per_gpu(self):
            return self.batch_size

        def tok_encode(self, string: str, **kwargs) -> List[int]:
            return self.tokenizer.encode(string, **kwargs)

        def tok_decode(self, tokens: List[int], **kwargs) -> str:
            return self.tokenizer.decode(tokens, **kwargs)

        def _model_call(self, inps: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
            """
            Performs a forward pass and returns logits.
            lm_eval expects (batch_size, seq_len, vocab_size) logits.
            """
            # inps and attention_mask are already on the correct device by lm_eval's internal batching
            with torch.no_grad():
                logits, _, _ = self.model(input_ids=inps, attention_mask=attention_mask)
            return logits

        def _model_generate(self, inps: torch.Tensor, gen_kwargs: Dict[str, Any]):
            """
            Generates text from prompts. Required for generation tasks (e.g., HumanEval).
            """
            # Pass through accelerator's unwrap_model to access the underlying model for generate if FSDP is used
            unwrapped_model = self.model.accelerator.unwrap_model(self.model) if hasattr(self.model, 'accelerator') else self.model
            
            with torch.no_grad():
                generated_ids = unwrapped_model.generate(
                    input_ids=inps,
                    max_length=gen_kwargs.get("max_length", self.max_length),
                    do_sample=gen_kwargs.get("do_sample", False),
                    temperature=gen_kwargs.get("temperature", 1.0),
                    top_p=gen_kwargs.get("top_p", 1.0),
                    num_beams=gen_kwargs.get("num_beams", 1),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=gen_kwargs.get("attention_mask", None)
                )
            return generated_ids[:, inps.shape[1]:] # Return only generated part


class Evaluator:
    """
    Manages all evaluation processes for the Gated Attention LLM, including PPL calculation,
    benchmark evaluation via lm_eval, and detailed internal metric analysis.
    """

    def __init__(self, model: GatedTransformer, data_loader: DataLoader, config: Config):
        """
        Initializes the Evaluator.

        Args:
            model: The GatedTransformer model to evaluate.
            data_loader: The DataLoader instance for evaluation data.
            config: The global configuration object.
        """
        self.model: GatedTransformer = model
        self.data_loader: DataLoader = data_loader
        self.config: Config = config

        self.accelerator: Accelerator = Accelerator(
            mixed_precision=self.config.training.mixed_precision,
            # We don't need logging for eval specific runs here, but it's consistent with Trainer
            log_with="tensorboard" if os.path.exists(os.path.join(self.config.output_dir, "runs")) else None,
            project_dir=self.config.output_dir
        )
        # Prepare model for distributed evaluation.
        # Note: If the model was already prepared by Trainer and passed, this might re-prepare.
        # A common pattern is to pass the already prepared model to Evaluator from Trainer.
        # For robustness, we prepare it here assuming it might be passed unprepared.
        self.model = self.accelerator.prepare(self.model)
        self.model.eval() # Ensure model is in eval mode after preparation

        # Conditional import and setup for lm_eval
        self.lm_eval_model = None
        if _LM_EVAL_AVAILABLE and (self.config.evaluation.benchmarks or self.config.evaluation.ruler_benchmark_enabled):
            # lm_eval's batch_size is usually per device.
            # Using the `global_batch_size` from training config, divided by number of processes.
            lm_eval_batch_size = self.config.training.global_batch_size // self.accelerator.num_processes
            # Ensure it's at least 1
            lm_eval_batch_size = max(1, lm_eval_batch_size)
            
            if self.accelerator.is_main_process:
                 print(f"Initializing lm_eval with batch size per device: {lm_eval_batch_size}")

            self.lm_eval_model = CustomLMEvalModel(
                model=self.model,
                tokenizer=self.data_loader.tokenizer,
                batch_size=lm_eval_batch_size,
                device=self.accelerator.device # Accelerator handles actual device
            )
        else:
            if self.accelerator.is_main_process:
                print("lm_eval benchmarks are disabled due to missing library or config settings.")


    def evaluate_ppl(self) -> float:
        """
        Calculates the perplexity (PPL) of the model on the evaluation dataset.

        Returns:
            The computed perplexity (float).
        """
        self.model.eval() # Ensure model is in evaluation mode
        
        eval_dataloader = self.data_loader.get_eval_dataloader()
        eval_dataloader = self.accelerator.prepare(eval_dataloader) # Prepare dataloader for distributed eval

        total_lm_loss_sum: torch.Tensor = torch.tensor(0.0, device=self.accelerator.device)
        total_active_tokens: torch.Tensor = torch.tensor(0, device=self.accelerator.device, dtype=torch.long)

        progress_bar = tqdm(
            eval_dataloader,
            desc="Evaluating PPL",
            disable=not self.accelerator.is_main_process
        )

        with torch.no_grad():
            for batch in progress_bar:
                input_ids, attention_mask, labels = batch # Unpack tuple if from TensorDataset

                # Forward pass - only language modeling loss is typically used for PPL
                # The model's forward returns lm_loss as the mean per active token
                _, lm_loss_per_token_mean, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                if lm_loss_per_token_mean is not None:
                    # Calculate number of active tokens in the batch (labels != -100)
                    num_batch_active_tokens = (labels != -100).sum()
                    
                    # Accumulate sum of losses (loss_per_token_mean * num_active_tokens)
                    total_lm_loss_sum += lm_loss_per_token_mean * num_batch_active_tokens
                    total_active_tokens += num_batch_active_tokens
                
        # Gather total loss sum and active tokens from all processes
        all_total_lm_loss_sum = self.accelerator.gather(total_lm_loss_sum).sum()
        all_total_active_tokens = self.accelerator.gather(total_active_tokens).sum()

        ppl: float
        if all_total_active_tokens > 0:
            avg_lm_loss = all_total_lm_loss_sum / all_total_active_tokens
            ppl = torch.exp(avg_lm_loss).item()
        else:
            ppl = float('inf') # No active tokens to evaluate

        self.model.train() # Restore model to training mode
        return ppl


    def evaluate_benchmarks(self) -> Dict[str, float]:
        """
        Performs few-shot evaluations on standard NLP benchmarks using the `lm_eval` library.

        Returns:
            A dictionary of benchmark results.
        """
        if not _LM_EVAL_AVAILABLE:
            if self.accelerator.is_main_process:
                print("lm_eval is not available, skipping benchmark evaluation.")
            return {"error": "lm_eval not available"}
        
        if self.lm_eval_model is None:
            if self.accelerator.is_main_process:
                print("lm_eval model wrapper not initialized, skipping benchmark evaluation.")
            return {"error": "lm_eval model wrapper not initialized"}

        if not (self.config.evaluation.benchmarks or self.config.evaluation.ruler_benchmark_enabled):
            if self.accelerator.is_main_process:
                print("No benchmarks specified in config, skipping benchmark evaluation.")
            return {}

        tasks = list(self.config.evaluation.benchmarks)
        if self.config.evaluation.ruler_benchmark_enabled:
            # Assuming 'ruler' is a task name recognized by lm_eval or a custom task.
            # If lm_eval does not have a built-in 'ruler', it might fail here.
            # A custom lm_eval task for RULER would be implemented separately.
            tasks.append("ruler") 
        
        # Filter out duplicates if any
        tasks = list(set(tasks))

        if self.accelerator.is_main_process:
            print(f"Running lm_eval benchmarks for tasks: {tasks}")

        # lm_eval.evaluator.evaluate function takes care of distributed execution
        # when an Accelerator is used internally by the LM (like CustomLMEvalModel).
        results = evaluator.evaluate(
            self.lm_eval_model,
            tasks,
            num_fewshot=0, # Default to 0-shot as few-shot can be very expensive. Can be configured.
            batch_size=self.lm_eval_model.batch_size_per_gpu,
            device=str(self.accelerator.device), # Pass device for lm_eval initialization
            no_cache=True, # Disable caching for consistent results
            limit=None, # Process all samples
            # output_path=os.path.join(self.config.output_dir, "lm_eval_results.json") # lm_eval saves this
        )
        
        if self.accelerator.is_main_process:
            print("lm_eval benchmark results:")
            print(json.dumps(results, indent=2))

        # Extract primary scores. The exact key depends on the benchmark.
        benchmark_scores: Dict[str, float] = {}
        for task_name, task_results in results['results'].items():
            if 'acc_norm' in task_results: # Normalized accuracy (e.g., MMLU, Hellaswag)
                benchmark_scores[task_name] = task_results['acc_norm'] * 100 # Convert to percentage
            elif 'acc' in task_results: # Accuracy
                benchmark_scores[task_name] = task_results['acc'] * 100
            elif 'perplexity' in task_results: # Perplexity tasks
                benchmark_scores[task_name] = task_results['perplexity']
            # Other metrics might be needed based on specific task outputs.
            # For HumanEval, often 'pass@1', 'pass@k'
            elif 'pass@1' in task_results:
                benchmark_scores[task_name] = task_results['pass@1'] * 100

        return benchmark_scores


    def analyze_metrics(self) -> Dict[str, Any]:
        """
        Conducts detailed analysis of gating scores, attention sinks, and massive activations
        as described in the paper's analysis sections.

        Returns:
            A dictionary containing aggregated analysis results.
        """
        self.model.eval() # Set model to evaluation mode
        
        eval_dataloader = self.data_loader.get_eval_dataloader()
        eval_dataloader = self.accelerator.prepare(eval_dataloader) # Prepare dataloader for distributed eval

        # Initialize lists to collect metrics across batches and layers
        # Each list will contain lists of tensors, one inner list per layer
        all_gating_scores_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.model.num_layers)]
        all_attention_weights_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.model.num_layers)]
        all_massive_activations_ffn_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.model.num_layers)]
        all_massive_activations_attn_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.model.num_layers)]

        # Limit analysis to a few batches to manage memory and computational cost
        # The paper implies analysis on 'test language modeling data'.
        # Let's process a reasonable number of batches, e.g., 20-50, or a full eval set if small enough.
        num_analysis_batches = min(len(eval_dataloader), 50) # Process max 50 batches for analysis

        progress_bar = tqdm(
            eval_dataloader,
            total=num_analysis_batches,
            desc="Analyzing Metrics",
            disable=not self.accelerator.is_main_process
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                if batch_idx >= num_analysis_batches:
                    break

                input_ids, attention_mask, _ = batch # Labels are not needed for metric collection

                # get_gating_metrics will run forward pass and collect internal metrics
                metrics_data = self.model.get_gating_metrics(input_ids=input_ids, attention_mask=attention_mask)

                # Aggregate metrics data per layer
                for layer_idx in range(self.model.num_layers):
                    if self.config.gating_enabled and self.config.evaluation.gating_score_analysis_enabled:
                        # Gating scores are returned as a list, one tensor per layer
                        if layer_idx < len(metrics_data['gating_scores']) and metrics_data['gating_scores'][layer_idx] is not None:
                            all_gating_scores_per_layer[layer_idx].append(metrics_data['gating_scores'][layer_idx])

                    if self.config.evaluation.attention_sink_analysis_enabled:
                        if layer_idx < len(metrics_data['attention_weights']) and metrics_data['attention_weights'][layer_idx] is not None:
                            all_attention_weights_per_layer[layer_idx].append(metrics_data['attention_weights'][layer_idx])

                    if self.config.evaluation.massive_activation_analysis_enabled:
                        if layer_idx < len(metrics_data['massive_activations_ffn']) and metrics_data['massive_activations_ffn'][layer_idx] is not None:
                            all_massive_activations_ffn_per_layer[layer_idx].append(metrics_data['massive_activations_ffn'][layer_idx])
                        if layer_idx < len(metrics_data['massive_activations_attn']) and metrics_data['massive_activations_attn'][layer_idx] is not None:
                            all_massive_activations_attn_per_layer[layer_idx].append(metrics_data['massive_activations_attn'][layer_idx])
        
        # Final aggregated results dictionary
        analysis_results: Dict[str, Any] = {}

        # 1. Gating Scores Analysis
        if self.config.gating_enabled and self.config.evaluation.gating_score_analysis_enabled:
            mean_gating_scores_by_layer = []
            for layer_idx, layer_scores in enumerate(all_gating_scores_per_layer):
                if layer_scores:
                    # Concatenate and compute mean
                    combined_scores = torch.cat(layer_scores, dim=0) # Shape: (Total_BS, SeqLen, NumHeads, HeadDim) or (Total_BS, SeqLen, NumHeads) etc.
                    mean_gating_scores_by_layer.append(self.accelerator.reduce(combined_scores.mean(), reduction="mean").item())
                else:
                    mean_gating_scores_by_layer.append(None) # No scores collected for this layer
            analysis_results["mean_gating_scores_by_layer"] = mean_gating_scores_by_layer
            if self.accelerator.is_main_process:
                print(f"Mean Gating Scores by Layer: {mean_gating_scores_by_layer}")

        # 2. Attention Sink (`F-Attn`) Analysis (Section 4.3, Figure 2, Table 4)
        if self.config.evaluation.attention_sink_analysis_enabled:
            f_attn_by_layer = []
            for layer_idx, layer_attn_weights in enumerate(all_attention_weights_per_layer):
                if layer_attn_weights:
                    # Concatenate all attention weights for the layer
                    combined_attn_weights = torch.cat(layer_attn_weights, dim=0) # (Total_B, H, S, S)
                    
                    # Extract attention to the first token (index 0 in the LAST sequence dimension S)
                    # This means for each query, how much attention is paid to the first key.
                    # This gives (Total_B, H, S_query) -> attention of each query to first key
                    first_token_scores_per_query = combined_attn_weights[..., 0] 
                    
                    # Average over batch, heads, and query positions for a single F-Attn score per layer
                    f_attn_score = self.accelerator.reduce(first_token_scores_per_query.mean(), reduction="mean").item()
                    f_attn_by_layer.append(f_attn_score)
                else:
                    f_attn_by_layer.append(None)
            analysis_results["f_attn_by_layer"] = f_attn_by_layer
            if self.accelerator.is_main_process:
                print(f"First Token Attention (F-Attn) by Layer: {f_attn_by_layer}")

        # 3. Massive Activations (`M-Act`) Analysis (Section 4.3, Table 4)
        if self.config.evaluation.massive_activation_analysis_enabled:
            m_act_ffn_by_layer = []
            m_act_attn_by_layer = []

            for layer_idx in range(self.model.num_layers):
                # FFN Activations
                if all_massive_activations_ffn_per_layer[layer_idx]:
                    combined_ffn_activations = torch.cat(all_massive_activations_ffn_per_layer[layer_idx], dim=0)
                    # Mean of maximum absolute activations across dimensions for the layer
                    m_act_ffn_by_layer.append(self.accelerator.reduce(combined_ffn_activations.abs().max(), reduction="mean").item())
                else:
                    m_act_ffn_by_layer.append(None)

                # Attention Activations (output of attention sub-layer before residual)
                if all_massive_activations_attn_per_layer[layer_idx]:
                    combined_attn_activations = torch.cat(all_massive_activations_attn_per_layer[layer_idx], dim=0)
                    # Mean of maximum absolute activations across dimensions for the layer
                    m_act_attn_by_layer.append(self.accelerator.reduce(combined_attn_activations.abs().max(), reduction="mean").item())
                else:
                    m_act_attn_by_layer.append(None)
            
            analysis_results["m_act_ffn_by_layer"] = m_act_ffn_by_layer
            analysis_results["m_act_attn_by_layer"] = m_act_attn_by_layer
            if self.accelerator.is_main_process:
                print(f"Max Absolute FFN Activations (M-Act FFN) by Layer: {m_act_ffn_by_layer}")
                print(f"Max Absolute Attention Activations (M-Act Attn) by Layer: {m_act_attn_by_layer}")

        self.model.train() # Restore model to training mode
        return analysis_results

