import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, GenerationConfig
from accelerate import Accelerator

from config import Config
from model.navil import NaViLModel
from dataset.multimodal_dataset import MultimodalDataset
from dataset.collate_fn import CustomCollateFn
from utils import logger


class NaViLEvaluator:
    """
    Manages the evaluation pipeline for the NaViL model across various benchmarks.
    This includes loading datasets, generating model responses, and computing metrics.
    """

    def __init__(self, model: NaViLModel, tokenizer: PreTrainedTokenizer, config: Config, accelerator: Optional[Accelerator] = None):
        """
        Initializes the NaViLEvaluator.

        Args:
            model: The trained NaViLModel instance.
            tokenizer: The pre-trained tokenizer for the LLM.
            config: The global configuration object.
            accelerator: Optional Hugging Face Accelerator for distributed evaluation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator

        # Set model to evaluation mode
        self.model.eval()

        # Determine device
        if self.accelerator:
            self.device = self.accelerator.device
            self.model = self.accelerator.prepare(self.model) # Prepare model for distributed inference
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        # Create output directory for evaluation results
        self.output_dir = self.config.get("evaluation.output_dir", "evaluation_results")
        if self.accelerator is None or self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Evaluation results will be saved to: {self.output_dir}")

        # Initialize CustomCollateFn for evaluation datasets
        self.collate_fn = CustomCollateFn(self.tokenizer, self.config)

        # Ensure special tokens are recognized (should be handled by MoELLM init already)
        _ = self.tokenizer.convert_tokens_to_ids(self.config.begin_of_image) # Just a check
        logger.info("NaViLEvaluator initialized. Model set to eval mode.")


    def evaluate(self, model_variant_name: str) -> Dict[str, Any]:
        """
        Orchestrates the evaluation process across all specified benchmarks.

        Args:
            model_variant_name: The name of the model variant being evaluated (e.g., 'navil_2b').

        Returns:
            A dictionary containing the aggregated results from all benchmarks.
        """
        all_results: Dict[str, Any] = {}
        benchmarks_to_run: List[str] = self.config.get("evaluation.benchmarks", [])
        
        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Starting evaluation for model variant: {model_variant_name}")
            logger.info(f"Benchmarks configured for evaluation: {benchmarks_to_run}")

        for benchmark_name in benchmarks_to_run:
            if self.accelerator is None or self.accelerator.is_main_process:
                logger.info(f"--- Running evaluation on benchmark: {benchmark_name} ---")

            # Dynamically dispatch to benchmark-specific evaluation methods
            method_name = f'_run_{benchmark_name.lower().replace("-", "_")}'
            if hasattr(self, method_name):
                try:
                    benchmark_results = getattr(self, method_name)()
                    all_results[benchmark_name] = benchmark_results
                    if self.accelerator is None or self.accelerator.is_main_process:
                        logger.info(f"Results for {benchmark_name}: {json.dumps(benchmark_results, indent=2)}")
                except NotImplementedError:
                    if self.accelerator is None or self.accelerator.is_main_process:
                        logger.warning(f"Benchmark '{benchmark_name}' evaluation not implemented. Skipping.")
                    all_results[benchmark_name] = {"error": "NotImplementedError"}
                except Exception as e:
                    if self.accelerator is None or self.accelerator.is_main_process:
                        logger.error(f"Error running benchmark '{benchmark_name}': {e}")
                    all_results[benchmark_name] = {"error": str(e)}
            else:
                if self.accelerator is None or self.accelerator.is_main_process:
                    logger.warning(f"No evaluation method found for benchmark '{benchmark_name}'. Skipping.")
                all_results[benchmark_name] = {"error": "MethodNotFound"}
        
        if self.accelerator is None or self.accelerator.is_main_process:
            # Save all results to a JSON file
            results_filepath = os.path.join(self.output_dir, f"{model_variant_name}_evaluation_results.json")
            with open(results_filepath, 'w') as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"All evaluation results saved to: {results_filepath}")
            logger.info("Evaluation process completed.")

        return all_results

    def _generate_response(
        self,
        image_tensors_list: List[torch.Tensor],
        prompt_text: str,
        generation_config_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generates a textual response from the NaViL model given an image and a text prompt.

        Args:
            image_tensors_list: A list of image tensors (different scales if VMP is active for evaluation).
                                Each tensor is (C, H, W). This comes directly from MultimodalDataset.__getitem__['image_tensors'].
            prompt_text: The input text prompt string.
            generation_config_dict: Optional dictionary of generation parameters (e.g., max_new_tokens, temperature).

        Returns:
            The decoded generated text string.
        """
        # Ensure generation configuration is provided, use defaults if not.
        _default_generation_config = {
            "max_new_tokens": self.config.get("evaluation.max_new_tokens", 256),
            "do_sample": self.config.get("evaluation.do_sample", False),
            "temperature": self.config.get("evaluation.temperature", 0.7),
            "num_beams": self.config.get("evaluation.num_beams", 1),
            "top_k": self.config.get("evaluation.top_k", 50),
            "top_p": self.config.get("evaluation.top_p", 1.0),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        actual_generation_config = {**_default_generation_config, **(generation_config_dict or {})}
        gen_config = GenerationConfig(**actual_generation_config)

        with torch.no_grad():
            # 1. Prepare dummy MultimodalDataset sample for CustomCollateFn
            # This allows CustomCollateFn to handle padding, special tokens, and VMP logic.
            dummy_dataset_sample = {
                'image_tensors': image_tensors_list,
                'text_ids': self.tokenizer.encode(prompt_text, add_special_tokens=False),
                'original_text': prompt_text,
                'image_path': "" # Not needed for generation
            }
            # CustomCollateFn expects a list of samples (a batch)
            collated_batch = self.collate_fn([dummy_dataset_sample])
            
            # Move collated batch data to device
            pixel_values_batch = collated_batch['images'].to(self.device)
            input_ids_with_placeholders_batch = collated_batch['input_ids'].to(self.device)
            attention_mask_batch = collated_batch['attention_mask'].to(self.device)
            visual_token_lengths_batch = collated_batch['visual_token_lengths'].to(self.device)
            visual_token_start_indices_batch = collated_batch['visual_token_start_indices'].to(self.device)

            # 2. Process images through Visual Encoder and Connector
            visual_features, _, _ = self.model.visual_encoder(pixel_values_batch)
            projected_visual_features = self.model.connector(visual_features)

            # 3. Get multimodal input embeddings and modality IDs for the LLM
            # This encapsulates the splicing logic
            inputs_embeds, modality_ids, final_attention_mask = self.model.mmoe_llm.get_multimodal_input_embeddings(
                input_ids_with_placeholders=input_ids_with_placeholders_batch,
                projected_visual_features=projected_visual_features,
                visual_token_lengths=visual_token_lengths_batch,
                visual_token_start_indices=visual_token_start_indices_batch
            )
            
            # The attention mask from collate_fn only covers the `input_ids_with_placeholders_batch`.
            # `final_attention_mask` from `get_multimodal_input_embeddings` is needed.
            # Make sure it's the right shape for generation.
            # For autoregressive decoding, the attention mask should cover all current tokens.
            
            # Assuming 'inputs_embeds' is the actual starting input for generation
            # and it already includes the `bos_token_id` equivalent at the start if necessary,
            # or `generate` will prepend it if `inputs_embeds` are just the prompt.
            # Hugging Face's `generate` method, when given `inputs_embeds`, will start from these.

            # The `modality_ids` are passed as custom `kwargs` to the patched `model.forward`.
            # `model.mmoe_llm.base_llm` is the `AutoModelForCausalLM` instance with patched forward.
            
            generated_ids = self.model.mmoe_llm.base_llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=final_attention_mask, # Use the mask that matches inputs_embeds
                modality_ids=modality_ids, # Pass custom kwarg to patched forward
                generation_config=gen_config,
                **actual_generation_config # Pass additional generation params
            )

            # Decode the generated tokens, skipping special tokens
            # generated_ids will include the initial prompt tokens
            # We want only the new generated text
            response_ids = generated_ids[:, inputs_embeds.shape[1]:] # Take only newly generated tokens
            generated_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
            
        return generated_text

    # --- Benchmark-Specific Evaluation Methods (Placeholders) ---
    # These methods simulate interaction with benchmarks and return dummy results.
    # Actual implementation would involve loading specific datasets, running inference,
    # and calculating metrics using official benchmark scripts/APIs.

    def _load_benchmark_dataset(self, benchmark_name: str) -> Dataset:
        """
        Placeholder function to load a specific benchmark dataset.
        In a real scenario, this would load data specific to MMVet, MMMU, etc.
        For now, it returns a dummy dataset.
        """
        class DummyBenchmarkDataset(Dataset):
            def __init__(self, num_samples: int = 10, img_size: int = 224, prompt: str = "Describe the image."):
                self.num_samples = num_samples
                self.img_size = img_size
                self.prompt = prompt
                # Generate dummy data
                self.data = []
                for i in range(num_samples):
                    # Simulate scaled image tensors: one scale only for simplicity here
                    dummy_img_tensor = torch.randn(3, img_size, img_size) # (C, H, W)
                    self.data.append({
                        'image_tensors': [dummy_img_tensor],
                        'text_ids': [], # Tokenized by collate_fn from prompt_text
                        'original_text': prompt,
                        'image_path': f"dummy_image_{i}.png",
                        'ground_truth_answer': f"This is a dummy answer for image {i} and prompt '{prompt}'.",
                    })

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.data[idx]

        logger.warning(f"Loading dummy dataset for benchmark '{benchmark_name}'. "
                       "Actual dataset loading and parsing logic needs to be implemented.")
        
        # This is a very simplified dummy dataset. A real implementation would parse
        # image paths, corresponding ground truths, and prompts from specific benchmark data formats.
        return DummyBenchmarkDataset(num_samples=10, prompt=f"Answer the question for {benchmark_name}.")

    def _run_mmvet(self) -> Dict[str, float]:
        """Placeholder for MMVet evaluation."""
        logger.info("Evaluating MMVet... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("MMVet")
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=self.collate_fn) # Batch size 1 for generation
        
        correct_predictions = 0
        total_samples = 0
        for i, batch in enumerate(dataloader):
            if i >= 5: break # Limit dummy run

            # The CustomCollateFn gives stacked images. For _generate_response, we need list of tensors.
            # This means we need to "unstack" `batch['images']` and pass the individual image tensors
            # that correspond to *one* logical image from the original `image_tensors_list` from `MultimodalDataset`.
            # This is complex with VMP.
            # For simplicity in this placeholder, assume single scale and directly use.
            
            # The dataset_item['image_tensors'] from MultimodalDataset is a LIST of scaled tensors.
            # The collate_fn stacks them. _generate_response expects a LIST.
            # So here, we are simulating a single item from a batch.
            # In a real scenario, we'd iterate over dataset_item['image_tensors'] from the `__getitem__`.
            # For the dummy dataset, `dataset[i]['image_tensors']` already provides this.
            
            image_tensors = dataset[i]['image_tensors'] # List of scaled tensors for one sample
            prompt_text = dataset[i]['original_text']
            ground_truth = dataset[i]['ground_truth_answer']
            
            response = self._generate_response(image_tensors, prompt_text)
            
            # Dummy metric: check if response contains a keyword from ground truth
            if "dummy answer" in response.lower() and str(i) in response: # Very naive check
                correct_predictions += 1
            total_samples += 1

            if self.accelerator is None or self.accelerator.is_main_process:
                logger.debug(f"MMVet Sample {i+1}: Prompt='{prompt_text[:50]}...', Response='{response[:50]}...', Ground Truth='{ground_truth[:50]}...'")

        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
        return {"accuracy": accuracy, "normalized_avg_score": accuracy}

    def _run_mmmu(self) -> Dict[str, float]:
        """Placeholder for MMMU evaluation."""
        logger.info("Evaluating MMMU... (Using dummy logic)")
        # This benchmark is more complex, involving multiple disciplines.
        # We'll return dummy scores.
        dataset = self._load_benchmark_dataset("MMMU")
        # Actual evaluation would involve generating responses and calling MMMU's eval script
        dummy_score = 40.0 + torch.rand(1).item() * 10 # Simulate a score between 40 and 50
        return {"avg_score": dummy_score, "normalized_avg_score": dummy_score}

    def _run_mmbench(self) -> Dict[str, float]:
        """Placeholder for MMBench evaluation."""
        logger.info("Evaluating MMBench... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("MMBench")
        dummy_score = 60.0 + torch.rand(1).item() * 15
        return {"score": dummy_score, "normalized_avg_score": dummy_score}

    def _run_mme(self) -> Dict[str, float]:
        """Placeholder for MME evaluation."""
        logger.info("Evaluating MME... (Using dummy logic)")
        # MME requires summing perception and cognition scores.
        perception_score = 1500 + torch.rand(1).item() * 200
        cognition_score = 300 + torch.rand(1).item() * 50
        total_score = perception_score + cognition_score
        # Paper normalizes to 0-100 for average. Need max possible score for normalization.
        # Max scores are not in paper. Assuming a simple mapping for "normalized_avg_score".
        normalized_score = (total_score / 2500) * 100 # Example max score of 2500 for normalization
        return {"perception": perception_score, "cognition": cognition_score, "total_score": total_score, "normalized_avg_score": normalized_score}

    def _run_mathvista(self) -> Dict[str, float]:
        """Placeholder for MathVista evaluation."""
        logger.info("Evaluating MathVista... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("MathVista")
        dummy_score = 30.0 + torch.rand(1).item() * 15
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_ocrbench(self) -> Dict[str, float]:
        """Placeholder for OCRBench evaluation."""
        logger.info("Evaluating OCRBench... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("OCRBench")
        dummy_score = 700 + torch.rand(1).item() * 100
        normalized_score = (dummy_score / 1000) * 100 # Example max score of 1000
        return {"score": dummy_score, "normalized_avg_score": normalized_score}

    def _run_ccbench(self) -> Dict[str, float]:
        """Placeholder for CCBench evaluation."""
        logger.info("Evaluating CCBench... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("CCBench")
        dummy_score = 70.0 + torch.rand(1).item() * 10
        return {"score": dummy_score, "normalized_avg_score": dummy_score}

    def _run_textvqa(self) -> Dict[str, float]:
        """Placeholder for TextVQA evaluation."""
        logger.info("Evaluating TextVQA... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("TextVQA")
        dummy_score = 70.0 + torch.rand(1).item() * 10
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_scienceqa_img(self) -> Dict[str, float]:
        """Placeholder for ScienceQA-IMG evaluation."""
        logger.info("Evaluating ScienceQA-IMG... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("ScienceQA-IMG")
        dummy_score = 90.0 + torch.rand(1).item() * 5
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_gqa(self) -> Dict[str, float]:
        """Placeholder for GQA evaluation."""
        logger.info("Evaluating GQA... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("GQA")
        dummy_score = 60.0 + torch.rand(1).item() * 5
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_docvqa(self) -> Dict[str, float]:
        """Placeholder for DocVQA evaluation."""
        logger.info("Evaluating DocVQA... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("DocVQA")
        dummy_score = 80.0 + torch.rand(1).item() * 10
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_ai2d(self) -> Dict[str, float]:
        """Placeholder for AI2D evaluation."""
        logger.info("Evaluating AI2D... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("AI2D")
        dummy_score = 70.0 + torch.rand(1).item() * 10
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_chartqa(self) -> Dict[str, float]:
        """Placeholder for ChartQA evaluation."""
        logger.info("Evaluating ChartQA... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("ChartQA")
        dummy_score = 70.0 + torch.rand(1).item() * 10
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_infographicvqa(self) -> Dict[str, float]:
        """Placeholder for InfographicVQA evaluation."""
        logger.info("Evaluating InfographicVQA... (Using dummy logic)")
        dataset = self._load_benchmark_dataset("InfographicVQA")
        dummy_score = 50.0 + torch.rand(1).item() * 10
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_mmlu(self) -> Dict[str, float]:
        """Placeholder for MMLU (NLP) evaluation."""
        logger.info("Evaluating MMLU... (Using dummy logic, integration with OpenCompass needed)")
        # In a real setup, this would use OpenCompass's API or CLI.
        dummy_score = 75.0 + torch.rand(1).item() * 2
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_cmmlu(self) -> Dict[str, float]:
        """Placeholder for CMMLU (NLP) evaluation."""
        logger.info("Evaluating CMMLU... (Using dummy logic, integration with OpenCompass needed)")
        dummy_score = 70.0 + torch.rand(1).item() * 5
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

    def _run_math(self) -> Dict[str, float]:
        """Placeholder for MATH (NLP) evaluation."""
        logger.info("Evaluating MATH... (Using dummy logic, integration with OpenCompass needed)")
        dummy_score = 60.0 + torch.rand(1).item() * 10
        return {"accuracy": dummy_score, "normalized_avg_score": dummy_score}

