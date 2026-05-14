import torch
from marlhf.ma_rlhf import MARLHF
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Configuration parameters
    policy_model_name = "gpt2"  # Smaller model for demonstration
    reward_model_name = "gpt2"  # Same for simplicity
    n_macro_action_length = 4
    clip_epsilon = 0.2
    gamma = 0.99
    lambda_gae = 0.95
    lr_policy = 1e-5
    lr_value = 1e-5
    num_epochs = 2
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing MA-RLHF agent...")
    ma_rlhf_agent = MARLHF(
        policy_model_name=policy_model_name,
        reward_model_name=reward_model_name,
        n_macro_action_length=n_macro_action_length,
        clip_epsilon=clip_epsilon,
        gamma=gamma,
        lambda_gae=lambda_gae,
        lr_policy=lr_policy,
        lr_value=lr_value
    )
    
    # Move models to the selected device
    ma_rlhf_agent.policy_lm.to(device)
    ma_rlhf_agent.policy_lm_ref.to(device)
    ma_rlhf_agent.value_head.to(device)
    ma_rlhf_agent.reward_model.to(device) # Though reward_model is a dummy here

    # Example prompts
    prompts = [
        "Tell me a short story about a brave knight.",
        "Write a Python function to reverse a string.",
        "Explain the concept of quantum entanglement in simple terms."
    ]

    # Load a dummy SFT model (can be the same as policy_lm_ref for initial testing)
    print("Loading SFT model (using policy_lm_ref as a placeholder)...")
    sft_model = AutoModelForCausalLM.from_pretrained(policy_model_name)
    sft_model.eval() # SFT model should be in eval mode
    sft_model.to(device)

    print("Starting MA-RLHF training...")
    ma_rlhf_agent.run_training(prompts=prompts, num_epochs=num_epochs, sft_model=sft_model)
    print("MA-RLHF training finished.")

    # Optional: Demonstrate generation after training
    print("
--- Demonstrating generation after training ---")
    test_prompt = "Once upon a time in a faraway land,"
    print(f"Prompt: {test_prompt}")
    prompt_tokens = ma_rlhf_agent.tokenizer.encode(test_prompt, return_tensors="pt").squeeze(0).tolist()
    generated_tokens, _ = ma_rlhf_agent.generate_sequence(prompt_tokens, max_new_tokens=50, temperature=0.7)
    generated_text = ma_rlhf_agent.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated response: {generated_text}")

if __name__ == "__main__":
    main()
