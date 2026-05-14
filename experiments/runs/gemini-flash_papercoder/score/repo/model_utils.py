import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from config import Config

logger = logging.getLogger(__name__)


class LLMForSelfCorrection:
    """
    A wrapper class for Hugging Face AutoModelForCausalLM, handling model loading,
    PEFT integration, text generation, and log-probability extraction.
    """

    def __init__(self, config: Config, is_ref_model: bool = False):
        """
        Initializes the language model wrapper.

        Args:
            config: An instance of the Config class containing hyperparameters and model details.
            is_ref_model: If True, this instance acts as the fixed reference model.
                          If False, this is the trainable policy model, and it will
                          internally load a separate, fixed reference model.
        """
        self.config: Config = config
        self.is_ref_model: bool = is_ref_model

        # Initialize tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.tokenizer.pad_token_id is None:
                # Fallback if EOS token also not set, typically for some models
                self.tokenizer.pad_token_id = 0  # A common default for padding
                logger.warning(
                    "pad_token_id and eos_token_id not found, setting pad_token_id to 0."
                )

        # Determine device
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"Using device: {self.device}")

        # Determine torch_dtype for model loading
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # Load the base model
        self.model: Union[PreTrainedModel, PeftModel] = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,  # Required for some models like CodeLlama, Gemma
            # low_cpu_mem_usage=True, # Optional: can reduce memory usage on CPU, but might be slower
        )
        self.model.to(self.device)

        self.ref_model: Optional[PreTrainedModel] = None

        if not self.is_ref_model:  # This is the policy model instance
            # Apply PEFT if configured
            if config.use_peft:
                if config.peft_method.lower() != "lora":
                    raise ValueError(
                        f"Only 'lora' PEFT method is supported, but got {config.peft_method}"
                    )

                peft_config = LoraConfig(
                    r=config.peft_lora_r,
                    lora_alpha=config.peft_lora_alpha,
                    lora_dropout=config.peft_lora_dropout,
                    bias="none",  # Common setting for causal LMs
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, peft_config)
                logger.info(
                    f"PEFT (LoRA) applied to policy model. Trainable parameters: {self.model.print_trainable_parameters()}"
                )
            else:
                logger.info("PEFT not used for policy model.")

            # Load the internal reference model (fixed)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                # low_cpu_mem_usage=True,
            )
            self.ref_model.eval()  # Set to evaluation mode
            self.ref_model.requires_grad_(False)  # Disable gradients
            self.ref_model.to(self.device)
            logger.info("Internal reference model loaded and set to eval mode.")
        else:  # This instance IS the dedicated reference model wrapper
            self.model.eval()
            self.model.requires_grad_(False)
            logger.info(
                "This LLMForSelfCorrection instance is set up as a fixed reference model."
            )

    def generate(
        self, prompt: str, temperature: float, max_new_tokens: int
    ) -> Tuple[str, List[float]]:
        """
        Generates a text response from the model and extracts the log-probabilities
        of the generated tokens.

        Args:
            prompt: The input string to the model.
            temperature: Sampling temperature. 0.0 for greedy decoding.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            A tuple containing:
            - generated_text: The decoded string of the generated response.
            - log_probs_list: A list of float representing the log-probabilities
                              of each token in the generated_text.
        """
        # Ensure model is in eval mode for consistent generation.
        # Note: If training, model should be switched to train() before optimizer step.
        # This generate function is for collecting rollouts.
        # The main training loop will manage model.train() and model.eval() appropriately.
        # We explicitly set to eval here to ensure no dropout/batchnorm effects during generation
        # if this function is called outside the context of a trainer's train() loop.
        current_model_state = self.model.training
        self.model.eval()

        with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                self.device
            )

            do_sample = temperature > 0.0

            generation_config = {
                "do_sample": do_sample,
                "temperature": temperature if do_sample else 1.0,  # Temperature > 1.0 if do_sample else 1.0
                "max_new_tokens": max_new_tokens,
                "return_dict_in_generate": True,
                "output_scores": True,  # Required to get logits for log-prob calculation
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # For greedy decoding (temperature=0), set `do_sample=False`
            if temperature == 0.0:
                generation_config["do_sample"] = False
                generation_config["num_beams"] = 1 # Required for greedy search

            output = self.model.generate(input_ids, **generation_config)

            # Extract generated token IDs (excluding input tokens)
            # output.sequences has shape (batch_size, sequence_length)
            # input_ids has shape (batch_size, prompt_length)
            generated_ids = output.sequences[:, input_ids.shape[1] :]

            # Decode generated text
            generated_text = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            # Calculate transition scores (log-probabilities) for generated tokens
            # compute_transition_scores returns a tensor of shape (batch_size, sequence_length - 1)
            # where sequence_length is the length of output.sequences.
            transition_scores = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )

            # Filter transition_scores to only include those for the *newly generated* tokens
            # The length of generated_ids[0] gives the count of new tokens.
            # transition_scores corresponds to these new tokens.
            log_probs_list = transition_scores[0, : generated_ids.shape[1]].tolist()

        # Restore model state
        if current_model_state:
            self.model.train()
        
        return generated_text, log_probs_list

    def get_log_probs(self, prompt: str, response: str) -> List[float]:
        """
        Calculates the log-probabilities of the given `response` tokens,
        conditioned on the `prompt`, using the model of this instance (`self.model`).

        Args:
            prompt: The input string that precedes the response.
            response: The generated response string for which log-probabilities are needed.

        Returns:
            A list of float representing the log-probabilities of each token in the `response`.
        """
        # Ensure model is in eval mode and no gradient computation
        current_model_state = self.model.training
        self.model.eval()

        with torch.no_grad():
            full_text = prompt + response

            # Tokenize full text to get input_ids and attention_mask for the forward pass
            inputs = self.tokenizer(
                full_text, return_tensors="pt", return_attention_mask=True
            ).to(self.device)

            # Tokenize prompt separately to determine its length in tokens
            prompt_tokenized = self.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_tokenized.shape[1]

            # Perform forward pass to get logits
            outputs = self.model(**inputs)
            # logits shape: (batch_size, sequence_length, vocab_size)
            logits = outputs.logits

            # Apply log_softmax to get log-probabilities over the vocabulary
            log_probs = torch.log_softmax(logits, dim=-1)

            response_log_probs: List[float] = []

            # Iterate through the tokens that constitute the 'response' part of the full_text.
            # Logits[0, i, :] predicts inputs.input_ids[0, i+1].
            # So, for the first response token (at index `prompt_len`), its log-prob is predicted
            # by logits at index `prompt_len - 1`.
            for i in range(
                prompt_len - 1, inputs.input_ids.shape[1] - 1
            ):  # Loop up to the token *before* the last token predicted
                token_id_to_predict = inputs.input_ids[0, i + 1]
                log_prob_for_token = log_probs[0, i, token_id_to_predict].item()
                response_log_probs.append(log_prob_for_token)
        
        # Restore model state
        if current_model_state:
            self.model.train()

        return response_log_probs

    def save_pretrained(self, path: str):
        """
        Saves the current state of the model and its tokenizer.
        If PEFT is enabled for a policy model, only the adapters are saved.

        Args:
            path: Directory where the model should be saved.
        """
        os.makedirs(path, exist_ok=True)
        if not self.is_ref_model and self.config.use_peft and isinstance(self.model, PeftModel):
            logger.info(f"Saving PEFT adapters to {path}")
            self.model.save_pretrained(path)
        else:
            logger.info(f"Saving full model to {path}")
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Tokenizer saved to {path}")

    def load_pretrained(self, path: str):
        """
        Loads a previously saved model state into this instance.

        Args:
            path: Directory from which to load the model.
        """
        torch_dtype = (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )  # Consistent dtype

        if not self.is_ref_model and self.config.use_peft:
            logger.info(f"Loading PEFT adapters from {path} into base model.")
            # Ensure base model is loaded first if this is a policy model with PEFT
            if not isinstance(self.model, PeftModel):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
                self.model.to(self.device)
            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            logger.info(f"Loading full model from {path}.")
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model and tokenizer loaded from {path}")

    def get_current_model(self) -> Union[PreTrainedModel, PeftModel]:
        """
        Returns the underlying Hugging Face model object for the current instance (policy or reference).
        """
        return self.model

    def get_ref_model(self) -> PreTrainedModel:
        """
        Returns the underlying Hugging Face reference model object.
        If this instance is itself the reference model wrapper (`is_ref_model` is True),
        it returns its own model (`self.model`). Otherwise, it returns the internally
        stored `self.ref_model` (which is the non-fine-tuned base model).
        """
        if self.is_ref_model:
            return self.model
        if self.ref_model is None:
            raise ValueError(
                "Reference model not initialized. This instance might be a policy model "
                "but ref_model was not loaded, or it was never meant to hold a ref_model. "
                "Ensure `is_ref_model` is True or `ref_model` is loaded internally."
            )
        return self.ref_model


if __name__ == "__main__":
    # Configure logging for better output during testing
    logging.basicConfig(level=logging.INFO)

    # Create a mock config object for testing
    class MockConfig(Config):
        def __init__(self):
            super().__init__()
            # Use a small, readily available model for quick testing
            self.base_model_name = "sshleifer/tiny-gpt2"
            self.use_peft = True
            self.peft_method = "lora"
            self.peft_lora_r = 8
            self.peft_lora_alpha = 16
            self.peft_lora_dropout = 0.1
            self.checkpoint_dir = "test_checkpoints"

    print("--- Initializing MockConfig ---")
    mock_config = MockConfig()
    os.makedirs(mock_config.checkpoint_dir, exist_ok=True)

    # --- Test 1: Policy Model Initialization (with PEFT) ---
    print("\n--- Test 1: Policy Model Initialization (with PEFT) ---")
    try:
        policy_model_wrapper = LLMForSelfCorrection(mock_config, is_ref_model=False)
        print("Policy model initialized successfully.")
        print(f"Policy model device: {policy_model_wrapper.device}")
        print(f"Policy model is PEFT model: {isinstance(policy_model_wrapper.model, PeftModel)}")
        print(f"Reference model exists: {policy_model_wrapper.ref_model is not None}")
        print(f"Reference model device: {policy_model_wrapper.ref_model.device}")
        assert isinstance(policy_model_wrapper.model, PeftModel)
        assert policy_model_wrapper.ref_model is not None
    except Exception as e:
        print(f"Error in Policy Model Initialization: {e}")

    # --- Test 2: Reference Model Initialization ---
    print("\n--- Test 2: Reference Model Initialization ---")
    try:
        ref_model_wrapper = LLMForSelfCorrection(mock_config, is_ref_model=True)
        print("Reference model wrapper initialized successfully.")
        print(f"Ref model is PEFT model: {isinstance(ref_model_wrapper.model, PeftModel)}")
        print(f"Internal ref model is None: {ref_model_wrapper.ref_model is None}")
        assert not isinstance(ref_model_wrapper.model, PeftModel) # Should be base model
        assert ref_model_wrapper.ref_model is None
    except Exception as e:
        print(f"Error in Reference Model Initialization: {e}")

    # --- Test 3: Generate Text and Log Probs ---
    print("\n--- Test 3: Generate Text and Log Probs ---")
    test_prompt = "Hello, my name is"
    max_gen_tokens = 5
    try:
        generated_text_greedy, log_probs_greedy = policy_model_wrapper.generate(
            test_prompt, temperature=0.0, max_new_tokens=max_gen_tokens
        )
        print(f"Greedy Generation:")
        print(f"  Prompt: '{test_prompt}'")
        print(f"  Generated Text: '{generated_text_greedy}'")
        print(f"  Log Probs (greedy): {log_probs_greedy}")
        assert len(log_probs_greedy) > 0 and isinstance(log_probs_greedy[0], float)

        generated_text_sample, log_probs_sample = policy_model_wrapper.generate(
            test_prompt, temperature=0.7, max_new_tokens=max_gen_tokens
        )
        print(f"Sampling Generation (temp=0.7):")
        print(f"  Prompt: '{test_prompt}'")
        print(f"  Generated Text: '{generated_text_sample}'")
        print(f"  Log Probs (sample): {log_probs_sample}")
        assert len(log_probs_sample) > 0 and isinstance(log_probs_sample[0], float)

    except Exception as e:
        print(f"Error in Generate Text: {e}")

    # --- Test 4: Get Log Probs for a given response ---
    print("\n--- Test 4: Get Log Probs for a given response ---")
    known_response = " Bob. I am a"
    try:
        full_log_probs = policy_model_wrapper.get_log_probs(
            test_prompt, known_response
        )
        ref_log_probs = ref_model_wrapper.get_log_probs(test_prompt, known_response)
        print(f"Get Log Probs for response '{known_response}':")
        print(f"  Policy Model Log Probs: {full_log_probs}")
        print(f"  Reference Model Log Probs: {ref_log_probs}")
        assert len(full_log_probs) > 0 and isinstance(full_log_probs[0], float)
        assert len(ref_log_probs) > 0 and isinstance(ref_log_probs[0], float)
        assert len(full_log_probs) == len(ref_log_probs)
    except Exception as e:
        print(f"Error in Get Log Probs: {e}")

    # --- Test 5: Save and Load Model (PEFT adapters) ---
    print("\n--- Test 5: Save and Load Model (PEFT adapters) ---")
    save_path_peft = os.path.join(mock_config.checkpoint_dir, "policy_peft_model")
    try:
        policy_model_wrapper.save_pretrained(save_path_peft)
        print(f"Policy model (PEFT) saved to {save_path_peft}")

        # Create a new wrapper to load into
        loaded_policy_model_wrapper = LLMForSelfCorrection(mock_config, is_ref_model=False)
        loaded_policy_model_wrapper.load_pretrained(save_path_peft)
        print("Policy model (PEFT) loaded successfully into new wrapper.")
        assert isinstance(loaded_policy_model_wrapper.model, PeftModel)
        # Verify a generation works
        generated_from_loaded, _ = loaded_policy_model_wrapper.generate(test_prompt, temperature=0.0, max_new_tokens=2)
        print(f"  Generated from loaded model: '{generated_from_loaded}'")
    except Exception as e:
        print(f"Error in Save/Load PEFT: {e}")

    # --- Test 6: Save and Load Model (Full Model, e.g., reference or non-PEFT policy) ---
    print("\n--- Test 6: Save and Load Model (Full Model) ---")
    save_path_full = os.path.join(mock_config.checkpoint_dir, "ref_full_model")
    try:
        ref_model_wrapper.save_pretrained(save_path_full)
        print(f"Reference model (full) saved to {save_path_full}")

        # Create a new wrapper to load into (as if it were a non-PEFT policy)
        mock_config_no_peft = MockConfig()
        mock_config_no_peft.use_peft = False
        loaded_full_model_wrapper = LLMForSelfCorrection(mock_config_no_peft, is_ref_model=False)
        loaded_full_model_wrapper.load_pretrained(save_path_full)
        print("Full model loaded successfully into new wrapper (non-PEFT).")
        assert not isinstance(loaded_full_model_wrapper.model, PeftModel)
        # Verify a generation works
        generated_from_loaded_full, _ = loaded_full_model_wrapper.generate(test_prompt, temperature=0.0, max_new_tokens=2)
        print(f"  Generated from loaded full model: '{generated_from_loaded_full}'")
    except Exception as e:
        print(f"Error in Save/Load Full Model: {e}")

    # --- Test 7: get_current_model and get_ref_model ---
    print("\n--- Test 7: get_current_model and get_ref_model ---")
    try:
        current_policy_model = policy_model_wrapper.get_current_model()
        current_ref_model_from_policy = policy_model_wrapper.get_ref_model()
        current_ref_model_from_ref_wrapper = ref_model_wrapper.get_current_model()

        print(f"Policy wrapper's current model: {type(current_policy_model)}")
        print(f"Policy wrapper's ref model: {type(current_ref_model_from_policy)}")
        print(f"Ref wrapper's current model: {type(current_ref_model_from_ref_wrapper)}")
        assert isinstance(current_policy_model, PeftModel)
        assert isinstance(current_ref_model_from_policy, PreTrainedModel)
        assert isinstance(current_ref_model_from_ref_wrapper, PreTrainedModel)
        print("get_current_model and get_ref_model methods work as expected.")
    except Exception as e:
        print(f"Error in model getters: {e}")

    print("\n--- All tests completed. ---")

    # Clean up test checkpoints
    import shutil
    if os.path.exists(mock_config.checkpoint_dir):
        shutil.rmtree(mock_config.checkpoint_dir)
        print(f"Cleaned up test directory: {mock_config.checkpoint_dir}")

