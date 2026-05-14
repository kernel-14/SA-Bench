
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
import math
import collections

from config import Config, SFTConfig, RMConfig, PPOConfig, MAConfig, ModelConfig, DataConfig, GeneralConfig
from model import PolicyModel, RewardModel, ValueModel
from data import get_dataloaders, DataCollatorForSFT, DataCollatorForRM, DataCollatorForPPO
from modules import MacroActionModule

class Trainer:
    def __init__(self, config: Config, task_name: str):
        self.config = config
        self.task_name = task_name
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.general.gradient_accumulation_steps,
            fp16=config.general.fp16,
            bf16=config.general.bf16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # For generation

        self.dataloaders = get_dataloaders(
            config.data, config.sft, config.rm, config.ppo, self.tokenizer, task_name
        )

        self.policy_model = PolicyModel(config.model)
        self.ref_policy_model = PolicyModel(config.model) # Reference model for KL divergence
        self.ref_policy_model.eval() # Keep reference model frozen

        if task_name != "apps":
            self.reward_model = RewardModel(config.model)
        else:
            self.reward_model = None # Reward is computed via compiler signal for APPS

        self.value_model = ValueModel(config.model)

        self.macro_action_module = MacroActionModule(config.ma, self.tokenizer)

        # Initialize optimizers and schedulers in SFT and RM stages (PPO has its own)
        self.sft_optimizer = None
        self.rm_optimizer = None
        self.sft_lr_scheduler = None
        self.rm_lr_scheduler = None

    def _get_optimizer_and_scheduler(self, model, lr, num_training_steps, warmup_ratio):
        optimizer = AdamW(model.parameters(), lr=lr)
        lr_scheduler = get_scheduler(
            name=self.config.sft.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=math.ceil(num_training_steps * warmup_ratio),
            num_training_steps=num_training_steps,
        )
        return optimizer, lr_scheduler

    def train_sft(self):
        self.accelerator.print("***** Starting SFT Training *****")
        sft_dataloader = self.dataloaders["sft_train"]
        
        self.sft_optimizer, self.sft_lr_scheduler = self._get_optimizer_and_scheduler(
            self.policy_model, 
            self.config.sft.learning_rate if self.task_name != "apps" else self.config.sft.code_learning_rate,
            len(sft_dataloader) * (self.config.sft.epochs if self.task_name != "apps" else self.config.sft.code_epochs),
            self.config.sft.warmup_ratio if self.task_name != "apps" else self.config.sft.code_warmup_ratio,
        )

        self.policy_model, self.sft_optimizer, self.sft_lr_scheduler, sft_dataloader = self.accelerator.prepare(
            self.policy_model, self.sft_optimizer, self.sft_lr_scheduler, sft_dataloader
        )

        self.policy_model.train()
        for epoch in range(self.config.sft.epochs if self.task_name != "apps" else self.config.sft.code_epochs):
            for step, batch in enumerate(tqdm(sft_dataloader, desc=f"SFT Epoch {epoch+1}")):
                with self.accelerator.accumulate(self.policy_model):
                    outputs = self.policy_model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.policy_model.parameters(), self.config.general.max_grad_norm)
                    self.sft_optimizer.step()
                    self.sft_lr_scheduler.step()
                    self.sft_optimizer.zero_grad()

                if (step + 1) % self.config.general.logging_steps == 0:
                    self.accelerator.print(f"SFT Epoch {epoch+1}, Step {step+1}: Loss = {loss.item():.4f}")

        self.accelerator.wait_for_everyone()
        self.policy_model.save_pretrained(os.path.join(self.config.general.output_dir, "sft_model"), save_function=self.accelerator.save)
        self.tokenizer.save_pretrained(os.path.join(self.config.general.output_dir, "sft_model"))
        self.accelerator.print("***** SFT Training Complete *****")

    def train_rm(self):
        if self.reward_model is None: # For APPS task, RM is skipped
            self.accelerator.print("***** Skipping RM Training for APPS task *****")
            return

        self.accelerator.print("***** Starting RM Training *****")
        rm_dataloader = self.dataloaders["rm_train"]
        rm_eval_dataloader = self.dataloaders["rm_eval"]

        self.rm_optimizer, self.rm_lr_scheduler = self._get_optimizer_and_scheduler(
            self.reward_model, 
            self.config.rm.learning_rate,
            len(rm_dataloader) * self.config.rm.epochs,
            self.config.rm.warmup_ratio,
        )
        
        self.reward_model, self.rm_optimizer, self.rm_lr_scheduler, rm_dataloader, rm_eval_dataloader = self.accelerator.prepare(
            self.reward_model, self.rm_optimizer, self.rm_lr_scheduler, rm_dataloader, rm_eval_dataloader
        )

        self.reward_model.train()
        best_eval_loss = float('inf')

        for epoch in range(self.config.rm.epochs):
            for step, batch in enumerate(tqdm(rm_dataloader, desc=f"RM Epoch {epoch+1}")):
                with self.accelerator.accumulate(self.reward_model):
                    chosen_rewards = self.reward_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                    rejected_rewards = self.reward_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
                    
                    # Reward Model loss: L_RM = -log(sigmoid(r(x, y+) - r(x, y-)))
                    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                    
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.reward_model.parameters(), self.config.general.max_grad_norm)
                    self.rm_optimizer.step()
                    self.rm_lr_scheduler.step()
                    self.rm_optimizer.zero_grad()

                if (step + 1) % self.config.general.logging_steps == 0:
                    self.accelerator.print(f"RM Epoch {epoch+1}, Step {step+1}: Loss = {loss.item():.4f}")
            
            # Evaluation
            eval_loss = self.evaluate_rm(rm_eval_dataloader)
            self.accelerator.print(f"RM Epoch {epoch+1} Evaluation Loss = {eval_loss:.4f}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.accelerator.wait_for_everyone()
                self.reward_model.save_pretrained(os.path.join(self.config.general.output_dir, "rm_model"), save_function=self.accelerator.save)
                self.tokenizer.save_pretrained(os.path.join(self.config.general.output_dir, "rm_model"))
                self.accelerator.print(f"New best RM model saved with eval loss: {best_eval_loss:.4f}")

        self.accelerator.print("***** RM Training Complete *****")

    def evaluate_rm(self, dataloader):
        self.reward_model.eval()
        total_loss = 0
        num_batches = 0
        for batch in tqdm(dataloader, desc="Evaluating RM"):
            with torch.no_grad():
                chosen_rewards = self.reward_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected_rewards = self.reward_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                total_loss += loss.item()
                num_batches += 1
        self.reward_model.train()
        return total_loss / num_batches

    def train_ppo(self):
        self.accelerator.print("***** Starting PPO Training *****")
        ppo_dataloader = self.dataloaders["ppo_train"]

        # Policy and Critic models have separate optimizers and schedulers
        policy_lr = self.config.ppo.policy_learning_rate if self.task_name != "apps" else self.config.ppo.code_policy_learning_rate
        critic_lr = self.config.ppo.critic_learning_rate if self.task_name != "apps" else self.config.ppo.code_critic_learning_rate
        ppo_warmup_steps = self.config.ppo.warmup_steps if self.task_name != "apps" else self.config.ppo.code_warmup_steps

        policy_optimizer = AdamW(self.policy_model.parameters(), lr=policy_lr)
        critic_optimizer = AdamW(self.value_model.parameters(), lr=critic_lr)

        num_training_steps = len(ppo_dataloader) * self.config.ppo.epochs
        policy_lr_scheduler = get_scheduler(
            name=self.config.sft.lr_scheduler_type, # Using SFT scheduler type
            optimizer=policy_optimizer,
            num_warmup_steps=ppo_warmup_steps,
            num_training_steps=num_training_steps,
        )
        critic_lr_scheduler = get_scheduler(
            name=self.config.sft.lr_scheduler_type,
            optimizer=critic_optimizer,
            num_warmup_steps=ppo_warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.policy_model, self.ref_policy_model, self.value_model, self.reward_model, \
        policy_optimizer, critic_optimizer, policy_lr_scheduler, critic_lr_scheduler, \
        ppo_dataloader = self.accelerator.prepare(
            self.policy_model, self.ref_policy_model, self.value_model, self.reward_model,
            policy_optimizer, critic_optimizer, policy_lr_scheduler, critic_lr_scheduler,
            ppo_dataloader
        )
        
        # Load SFT model for policy and reference, and RM model for reward
        if os.path.exists(os.path.join(self.config.general.output_dir, "sft_model")):
            self.policy_model.load_pretrained(os.path.join(self.config.general.output_dir, "sft_model"), is_accelerate_model=True)
            self.ref_policy_model.load_pretrained(os.path.join(self.config.general.output_dir, "sft_model"), is_accelerate_model=True)
        if self.reward_model and os.path.exists(os.path.join(self.config.general.output_dir, "rm_model")):
            self.reward_model.load_pretrained(os.path.join(self.config.general.output_dir, "rm_model"), is_accelerate_model=True)

        for epoch in range(self.config.ppo.epochs):
            for step, batch in enumerate(tqdm(ppo_dataloader, desc=f"PPO Epoch {epoch+1}")):
                self.policy_model.train()
                self.value_model.train()
                self.reward_model.eval() # Keep RM in eval mode

                with self.accelerator.accumulate(self.policy_model, self.value_model):
                    prompt_input_ids = batch["prompt_input_ids"]
                    prompt_attention_mask = batch["prompt_attention_mask"]
                    start_idx = prompt_input_ids.size(1)

                    # 1. Generate responses from the current policy model
                    generation_kwargs = {
                        "max_new_tokens": self.config.ppo.max_response_length,
                        "temperature": self.config.ppo.temperature,
                        "top_p": self.config.ppo.top_p,
                        "top_k": self.config.ppo.top_k if self.task_name != "apps" else self.config.ppo.code_top_k,
                        "do_sample": True,
                        "return_dict_in_generate": True,
                        "output_scores": True,
                    }
                    
                    policy_output = self.policy_model.generate(
                        prompt_input_ids,
                        prompt_attention_mask,
                        **generation_kwargs,
                    )
                    generated_sequence = policy_output.sequences
                    
                    # Compute logprobs for the generated sequence from policy and reference model
                    # The `generated_sequence` contains prompt + generated tokens.
                    # We need logprobs for only the generated part.
                    policy_logits = self.policy_model(generated_sequence, attention_mask=(generated_sequence != self.tokenizer.pad_token_id)).logits
                    ref_logits = self.ref_policy_model(generated_sequence, attention_mask=(generated_sequence != self.tokenizer.pad_token_id)).logits

                    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
                    ref_logprobs = F.log_softmax(ref_logits, dim=-1)

                    # Select logprobs for the generated tokens only
                    # This needs to be careful with padding and actual generated lengths
                    action_logprobs = torch.gather(policy_logprobs[:, :-1, :], dim=2, index=generated_sequence[:, 1:].unsqueeze(2)).squeeze(2)
                    ref_action_logprobs = torch.gather(ref_logprobs[:, :-1, :], dim=2, index=generated_sequence[:, 1:].unsqueeze(2)).squeeze(2)

                    # Mask out prompt tokens and padding for logprobs
                    attention_mask_generated = (generated_sequence[:, 1:] != self.tokenizer.pad_token_id) & (generated_sequence[:, 1:] != -100) # Assuming -100 is mask label
                    action_logprobs = action_logprobs * attention_mask_generated
                    ref_action_logprobs = ref_action_logprobs * attention_mask_generated

                    # Compute KL divergence for reward shaping
                    kl_div = (action_logprobs - ref_action_logprobs).sum(dim=-1) # Sum over generated tokens

                    # 2. Get rewards from Reward Model
                    if self.task_name != "apps":
                        rewards_from_rm = self.reward_model(generated_sequence, attention_mask=(generated_sequence != self.tokenizer.pad_token_id))
                    else:
                        # For APPS, rewards come from external compiler signal. Placeholder for now.
                        # This would typically be a function that executes the code and returns a scalar reward.
                        # For now, we'll use a dummy reward.
                        rewards_from_rm = torch.zeros(prompt_input_ids.size(0), 1, device=self.accelerator.device)
                        # In a real scenario, this would involve executing the code and getting a pass/fail
                        # as described in Appendix B.5
                        # R(x,y) = -0.3 + 1.3 * N_pass / (N_pass + N_fail) if compiled
                        #        = -0.6 if runtime error
                        #        = -1.0 if compile error
                        # This is a critical external component not directly implemented in this code structure.
                        # For reproduction, the environment would need to provide this.

                    # Reshape rewards for per-token or per-macro-action application
                    # The base reward is the RM score, which applies to the *entire* generated sequence.
                    # The KL penalty is also per sequence.
                    # R(x,y) = r_phi(x,y) - beta * D_KL(pi_theta || pi_sft)
                    
                    # For MA-RLHF, the reward R_tau is for the macro action.
                    # The paper implies that the final reward R(x,y) is distributed over tokens/macro_actions.
                    # "These advantage estimates and state-action value functions are then used to all tokens
                    # within the macro action during the optimization of both the policy and critic models."
                    # This means we need a way to assign the final sequence reward and KL to individual tokens or macro actions.

                    # One common approach: distribute the reward and KL penalty evenly or based on token importance.
                    # The paper says: "The macro reward for executing the macro action ωτ at the macro time step τ
                    # is defined as: Rτ = E[Σ_i=0^|ωτ|-1 ρ^i r_{tτ+i} | sτ], where rt is the reward received at t,
                    # and we set the discount factor ρ = 1 in our experiments."
                    # This means rt needs to be defined. rt = r_phi - beta * D_KL seems logical.
                    # But then R_tau is sum of token rewards.

                    # A simpler interpretation for single sequence reward: apply it at the end.
                    # However, PPO needs per-step rewards or value estimates.
                    # Let's assume the overall reward (RM - KL) is assigned to the last token of the full sequence.
                    # This makes sense if the reward model gives a score for the *entire* completion.

                    # Let's re-read the PPO part in appendix E:
                    # "Get reward score at current experience r := πrm(x, y);" (this is per sequence)
                    # "Compute maco actions {ωτ}=1 bas on the ermination rule {ωτ}r=1 := ζ(y);"
                    # "Compute macro action value function" (from token values)
                    # "Obtain Åτ and τ with GAE(V π(sτ, ωτ), r);"  <- This 'r' is the sequence reward.

                    # This suggests the sequence reward `r` needs to be transformed into macro action rewards `R_tau`.
                    # A common way to handle this in RLHF is to define a per-token reward, where most tokens
                    # get a KL penalty, and only the last token gets the RM reward.
                    
                    # Let's define `per_token_rewards` as zero everywhere except the last token of the generated sequence,
                    # which gets `rewards_from_rm - kl_coefficient * kl_div`.
                    # Then GAE can sum these up.

                    per_token_rewards = torch.zeros_like(action_logprobs, dtype=torch.float)
                    # Apply KL penalty to all tokens
                    per_token_rewards -= self.config.ppo.kl_coefficient * (action_logprobs - ref_action_logprobs)
                    
                    # Apply the RM reward to the last token of each generated sequence
                    # Find the length of each generated sequence (excluding prompt)
                    response_lengths = (generated_sequence[:, start_idx:] != self.tokenizer.pad_token_id).sum(dim=1)
                    
                    for i in range(prompt_input_ids.size(0)):
                        if response_lengths[i] > 0:
                            # Apply the full RM reward for the sequence to the last token of the response
                            # rewards_from_rm is (batch_size, 1), so use .squeeze(-1)
                            per_token_rewards[i, response_lengths[i] - 1] += rewards_from_rm[i].squeeze(-1)

                    # 3. Get values from Value Model
                    policy_values = self.value_model(generated_sequence, attention_mask=(generated_sequence != self.tokenizer.pad_token_id)).squeeze(-1)
                    old_policy_values = policy_values.detach() # Used for clipping in critic loss
                    
                    # Extract values for only the generated part
                    policy_values_generated = policy_values[:, start_idx:]
                    old_policy_values_generated = old_policy_values[:, start_idx:]


                    # 4. Compute macro actions and their values/rewards
                    batch_macro_action_positions = []
                    batch_macro_action_rewards = []
                    batch_macro_action_values = []
                    
                    for i in range(prompt_input_ids.size(0)): # Iterate over each sequence in the batch
                        current_generated_tokens = generated_sequence[i, start_idx:].tolist()
                        current_attention_mask = (generated_sequence[i] != self.tokenizer.pad_token_id).long()
                        current_per_token_rewards = per_token_rewards[i]
                        current_policy_values_generated = policy_values_generated[i].unsqueeze(0) # (1, num_generated_tokens)
                        current_old_policy_values_generated = old_policy_values_generated[i].unsqueeze(0) # (1, num_generated_tokens)
                        
                        # Get perplexity scores if needed for termination condition
                        ppl_scores = None # Placeholder. Requires more detailed PPL calculation.
                        if self.config.ma.termination_condition == "perplexity":
                            # This needs to be calculated from logits, typically from the reference model
                            # as stated in Appendix B.4 ("leverages the logits from the reference model")
                            # For now, it's not implemented, so the dummy ppl_scores will be ignored
                            # if termination_condition != "perplexity"
                            pass # TODO: Implement PPL score calculation

                        macro_action_positions = self.macro_action_module.get_macro_action_positions(
                            start_idx=0, # Relative to the generated part for slicing
                            attention_mask=current_attention_mask,
                            generated_tokens=current_generated_tokens,
                            ppl_scores=ppl_scores
                        )
                        # Adjust positions back to be relative to `start_idx`
                        macro_action_positions_abs = [pos + start_idx for pos in macro_action_positions]
                        batch_macro_action_positions.append(macro_action_positions_abs)

                        # Now convert per_token_rewards into macro_action_rewards
                        # This involves summing token rewards within each macro action.
                        macro_rewards_for_seq = []
                        current_ma_start_rel = 0
                        for ma_end_rel in macro_action_positions[1:]:
                            macro_rewards_for_seq.append(current_per_token_rewards[current_ma_start_rel : ma_end_rel].sum())
                            current_ma_start_rel = ma_end_rel
                        batch_macro_action_rewards.append(torch.stack(macro_rewards_for_seq))

                        # Get macro action values from token values
                        # The macro_action_module.get_macro_action_values expects values for the generated part
                        # and macro_action_positions relative to the start of that part.
                        macro_values_for_seq = self.macro_action_module.get_macro_action_values(
                            token_values=current_policy_values_generated.unsqueeze(-1), # (1, num_generated_tokens, 1)
                            attention_mask=current_attention_mask.unsqueeze(0), # (1, total_sequence_length)
                            start_idx=0, # Relative to current_policy_values_generated
                            macro_action_positions=macro_action_positions,
                        )
                        batch_macro_action_values.append(macro_values_for_seq.squeeze(0)) # Squeeze batch dim for list append
                    
                    # Pad macro action rewards and values to the same length for batching
                    # Find max num_macro_actions in batch
                    max_num_ma = max([len(ma_rewards) for ma_rewards in batch_macro_action_rewards])

                    padded_macro_action_rewards = []
                    padded_macro_action_values = []
                    for i in range(prompt_input_ids.size(0)):
                        ma_rewards = batch_macro_action_rewards[i]
                        ma_values = batch_macro_action_values[i]
                        
                        pad_len = max_num_ma - len(ma_rewards)
                        padded_rewards = F.pad(ma_rewards, (0, pad_len), "constant", 0.0)
                        padded_values = F.pad(ma_values, (0, pad_len), "constant", 0.0)
                        padded_macro_action_rewards.append(padded_rewards)
                        padded_macro_action_values.append(padded_values)
                    
                    padded_macro_action_rewards = torch.stack(padded_macro_action_rewards)
                    padded_macro_action_values = torch.stack(padded_macro_action_values)

                    # 5. Compute Advantages and Returns for macro actions
                    advantages, returns = self.macro_action_module.get_advantages_and_returns(
                        padded_macro_action_values,
                        padded_macro_action_rewards,
                        self.config.ppo.gamma,
                        self.config.ppo.lam,
                    )
                    
                    # 6. PPO Update Loop (Inner loop for PPO epochs)
                    for ppo_epoch in range(self.config.ppo.ppo_epochs):
                        # Policy Loss
                        policy_loss = self.macro_action_module.policy_loss_macro_action(
                            action_logprobs, # Logprobs of current policy for generated tokens
                            action_logprobs.detach(), # Old logprobs for ratio computation
                            advantages, # Macro advantages
                            attention_mask_generated, # Mask for generated tokens
                            batch_macro_action_positions[0], # Using first sequence's positions (assumes consistent structure)
                            self.config.ppo.clip_ratio,
                            start_idx=0, # Relative to the generated part
                        )

                        # Critic Loss
                        critic_loss = self.macro_action_module.critic_loss_macro_action(
                            policy_values_generated.unsqueeze(-1), # Current token values
                            old_policy_values_generated.unsqueeze(-1), # Old token values for clipping
                            returns, # Macro returns
                            attention_mask_generated, # Mask for generated tokens
                            batch_macro_action_positions[0], # Using first sequence's positions
                            self.config.ppo.clip_ratio,
                            start_idx=0, # Relative to the generated part
                        )
                        
                        total_loss = policy_loss + critic_loss
                        
                        self.accelerator.backward(total_loss)
                        self.accelerator.clip_grad_norm_(self.policy_model.parameters(), self.config.general.max_grad_norm)
                        self.accelerator.clip_grad_norm_(self.value_model.parameters(), self.config.general.max_grad_norm)

                        policy_optimizer.step()
                        critic_optimizer.step()
                        policy_lr_scheduler.step()
                        critic_lr_scheduler.step()

                        policy_optimizer.zero_grad()
                        critic_optimizer.zero_grad()

                    if (step + 1) % self.config.general.logging_steps == 0:
                        self.accelerator.print(f"PPO Epoch {epoch+1}, Step {step+1}: Policy Loss = {policy_loss.item():.4f}, Critic Loss = {critic_loss.item():.4f}, Total Reward = {rewards_from_rm.mean().item():.4f}")

        self.accelerator.wait_for_everyone()
        self.policy_model.save_pretrained(os.path.join(self.config.general.output_dir, "ppo_policy_model"), save_function=self.accelerator.save)
        self.tokenizer.save_pretrained(os.path.join(self.config.general.output_dir, "ppo_policy_model"))
        self.accelerator.print("***** PPO Training Complete *****")

    def run(self):
        # 1. SFT Stage
        self.train_sft()

        # 2. RM Stage
        self.train_rm()

        # 3. PPO Stage (RLHF)
        self.train_ppo()


if __name__ == "__main__":
    # Example usage:
    # Set up a dummy config for demonstration
    cfg = Config()
    cfg.model.model_name_or_path = "gpt2" # Using a small model for testing
    cfg.general.output_dir = "./outputs_test"
    cfg.general.logging_steps = 10
    cfg.sft.epochs = 1
    cfg.rm.epochs = 1
    cfg.ppo.epochs = 1
    cfg.sft.batch_size = 2 # Small batch size for testing
    cfg.rm.batch_size = 2
    cfg.ppo.batch_size = 2
    cfg.ppo.max_response_length = 20
    cfg.ppo.max_prompt_length = 30
    cfg.ma.n_gram = 3 # Fixed 3-gram for testing
    
    # Create output directory
    os.makedirs(cfg.general.output_dir, exist_ok=True)

    # Initialize and run trainer for a specific task (e.g., "tldr")
    # Note: Dataset loading will require actual data or mock data to run.
    # For this example, we'll run a minimal PPO with dummy data later to avoid
    # requiring actual datasets for testing the structure.
    
    # For now, let's just make sure the Trainer class initializes without errors.
    # To run this directly, you would need to set up mock datasets or
    # have the actual datasets accessible to `data.py`.
    
    # trainer = Trainer(cfg, task_name="tldr")
    # trainer.run()
    print("Trainer class defined. To run, instantiate and call .run() with appropriate task_name and dataset setup.")
    print("Example: trainer = Trainer(Config(), task_name='tldr'); trainer.run()")

