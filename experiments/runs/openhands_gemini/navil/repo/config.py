
import argparse

class NaViLConfig:
    def __init__(self, model_size="2B"):
        self.model_size = model_size
        self.visual_encoder = {}
        self.llm = {}
        self.training = {}
        self._set_config(model_size)

    def _set_config(self, model_size):
        if model_size == "2B":
            # NaViL-2B specific configurations (Table 6)
            self.visual_encoder = {
                "params": 0.6e9,
                "depth": 24,
                "width": 1472,
                "mlp_width": 5888,
                "attention_heads": 23,
                "patch_embedding_stride": 16, # Section 5.1
            }
            self.llm = {
                "init_model": "InternLM2-1.8B", # Section 5.1
                "activated_params": 1.8e9,
                "depth": 24,
                "width": 2048,
                "mlp_width": 8192,
                "attention_heads": 16,
                "num_experts": 2, # Section 3.2.2 and Table 6
                "attention_type": "causal", # Section 5.1
                "rope_type": "1D-RoPE", # Section 5.1
                "tokenizer": "InternLM2", # Section 5.1
                "conversation_format": "InternLM2", # Section 5.1
            }
            # Training hyperparameters (Table 7)
            self.training = {
                "max_image_patches": 4096,
                "llm_max_sequence_length": 16384,
                "optimizer": "AdamW",
                "optimizer_betas": (0.9, 0.95),
                "optimizer_eps": 1e-8,
                "gradient_accumulation_steps": 1, # Not explicitly stated, assuming 1 for simplicity if not mentioned
                "numerical_precision": "bfloat16",
                "visual_multi_scale_packing": True, # Section 4.1
                "downsampling_rate_tau": (2**0.5) / 2, # Section 4.1

                # Stage 1.1: Multi-modal Generative Pre-training (initial)
                "s1_1_training_steps": 70000,
                "s1_1_global_batch_size": 7000,
                "s1_1_weight_decay": 0.05,
                "s1_1_learning_rate_schedule": "constant_warmup",
                "s1_1_peak_learning_rate": 5e-5,
                "s1_1_warmup_steps": 200,
                "s1_1_data_sources": ["Laion-2B", "Coyo-700M", "Wukong", "SA-1B"], # Section 4.2
                "s1_1_data_mix_web_scale_count": 300e6, # Section 4.2
                "s1_1_data_mix_synthesized_count": 200e6, # Section 4.2
                "s1_1_frozen_text_params": True, # Section 4.2 (only vision-specific params trainable)

                # Stage 1.2: Multi-modal Generative Pre-training (fine-tuned)
                "s1_2_training_steps": 40000,
                "s1_2_global_batch_size": 4614,
                "s1_2_weight_decay": 0.1,
                "s1_2_learning_rate_schedule": "cosine_decay",
                "s1_2_peak_learning_rate": 5e-5,
                "s1_2_warmup_steps": 200,
                "s1_2_data_sources": ["InternVL-2.5 high-quality data", "InternLM2.5 pure language data"], # Section D
                "s1_2_data_count": 185e6, # Section 4.2
                "s1_2_frozen_text_params": False, # Section 4.2 (self-attention layers unfrozen)

                # Stage 2: Supervised Fine-tuning
                "s2_training_steps": 30000,
                "s2_global_batch_size": 2234, # Assuming this from table 7, row 4, column 4
                "s2_weight_decay": 0.01,
                "s2_learning_rate_schedule": "cosine_decay",
                "s2_peak_learning_rate": 2e-5,
                "s2_warmup_steps": 200,
                "s2_data_sources": ["InternVL-2.5 high-quality data"], # Section D
                "s2_data_count": 68e6, # Section 4.2
                "s2_frozen_text_params": False, # Section 4.2 (all parameters unfrozen)
            }
        elif model_size == "9B":
            # NaViL-9B specific configurations (Table 6)
            self.visual_encoder = {
                "params": 1.2e9,
                "depth": 32,
                "width": 1792,
                "mlp_width": 7168,
                "attention_heads": 28,
                "patch_embedding_stride": 16,
            }
            self.llm = {
                "init_model": "Qwen3-8B", # Section 5.1
                "activated_params": 8.0e9,
                "depth": 36,
                "width": 4096,
                "mlp_width": 12288,
                "attention_heads": 32,
                "num_experts": 2,
                "attention_type": "causal",
                "rope_type": "1D-RoPE",
                "tokenizer": "Qwen3",
                "conversation_format": "Qwen3",
            }
            # Training hyperparameters (Table 8)
            self.training = {
                "max_image_patches": 4096,
                "llm_max_sequence_length": 16384,
                "optimizer": "AdamW",
                "optimizer_betas": (0.9, 0.95),
                "optimizer_eps": 1e-8,
                "gradient_accumulation_steps": 1, # Not explicitly stated
                "numerical_precision": "bfloat16",
                "visual_multi_scale_packing": True,
                "downsampling_rate_tau": (2**0.5) / 2,

                # Stage 1.1: Multi-modal Generative Pre-training (initial)
                "s1_1_training_steps": 50000,
                "s1_1_global_batch_size": 1792,
                "s1_1_weight_decay": 0.05,
                "s1_1_learning_rate_schedule": "constant_warmup",
                "s1_1_peak_learning_rate": 5e-5,
                "s1_1_warmup_steps": 200,
                "s1_1_data_sources": ["Laion-2B", "Coyo-700M", "Wukong", "SA-1B"],
                "s1_1_data_mix_web_scale_count": 300e6,
                "s1_1_data_mix_synthesized_count": 200e6,
                "s1_1_frozen_text_params": True,
                "s1_1_visual_multi_scale_packing_enabled": False, # Section A

                # Stage 1.2: Multi-modal Generative Pre-training (fine-tuned)
                "s1_2_training_steps": 33000,
                "s1_2_global_batch_size": 3520,
                "s1_2_weight_decay": 0.1,
                "s1_2_learning_rate_schedule": "cosine_decay",
                "s1_2_peak_learning_rate": 5e-5,
                "s1_2_warmup_steps": 200,
                "s1_2_data_sources": ["InternVL-2.5 high-quality data", "InternLM2.5 pure language data"],
                "s1_2_data_count": 185e6,
                "s1_2_frozen_text_params": False,
                "s1_2_visual_multi_scale_packing_enabled": True,

                # Stage 2: Supervised Fine-tuning
                "s2_training_steps": 6000,
                "s2_global_batch_size": 2234, # Assuming this from table 8, row 4, column 4
                "s2_weight_decay": 0.01,
                "s2_learning_rate_schedule": "cosine_decay",
                "s2_peak_learning_rate": 2e-5,
                "s2_warmup_steps": 200,
                "s2_data_sources": ["InternVL-2.5 high-quality data"],
                "s2_data_count": 68e6,
                "s2_frozen_text_params": False,
                "s2_visual_multi_scale_packing_enabled": True,
            }
        else:
            raise ValueError(f"Unknown model size: {model_size}. Choose '2B' or '9B'.")

    def __str__(self):
        return f"NaViL Configuration ({self.model_size}):\n" \
               f"  Visual Encoder: {self.visual_encoder}\n" \
               f"  LLM: {self.llm}\n" \
               f"  Training: {self.training}"

def get_config(model_size="2B"):
    return NaViLConfig(model_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NaViL Configuration")
    parser.add_argument("--model_size", type=str, default="2B", choices=["2B", "9B"],
                        help="Specify the NaViL model size (e.g., '2B', '9B').")
    args = parser.parse_args()

    config = get_config(args.model_size)
    print(config)
