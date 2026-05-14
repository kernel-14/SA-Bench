
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import config
from typing import List

class LLM:
    def __init__(self, model_name: str, load_in_4bit: bool = True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Use a dedicated pad token if not available

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto", # Automatically distributes the model across available devices
        )
        # Resize token embeddings if a new pad token was added
        if self.tokenizer.pad_token is not None and self.tokenizer.pad_token == '[PAD]' and \
           len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.eval() # Set to evaluation mode by default

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0, num_return_sequences: int = 1) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(config.DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only the newly generated tokens
        generated_sequences = []
        for i in range(num_return_sequences):
            # The output contains the prompt tokens as well, so we slice them off
            generated_text = self.tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            generated_sequences.append(generated_text)
            
        return generated_sequences

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

