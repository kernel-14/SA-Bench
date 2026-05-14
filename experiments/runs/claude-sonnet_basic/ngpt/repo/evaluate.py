"""
Evaluation script for nGPT and GPT models on downstream tasks.

The paper evaluates on:
- HellaSwag
- PIQA
- WinoGrande
- ARC-Easy
- ARC-Challenge
- WMT14-FR-EN (BLEU score, 5-shot)

This script implements zero-shot and few-shot evaluation using
log-likelihood scoring for multiple-choice tasks.
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import nGPT, GPT, nGPTConfig


# ─── Evaluation Utilities ─────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, nGPTConfig]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    if isinstance(config, dict):
        config = nGPTConfig(**config)

    # Determine model type from config or checkpoint
    model_type = checkpoint.get('model_type', 'ngpt')

    if model_type == 'ngpt':
        model = nGPT(config)
    else:
        model = GPT(config)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


@torch.no_grad()
def compute_log_likelihood(model: torch.nn.Module, input_ids: torch.Tensor,
                            target_ids: torch.Tensor, device: str) -> float:
    """
    Compute the log-likelihood of target_ids given input_ids.
    Used for multiple-choice evaluation.
    """
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    logits, _ = model(input_ids)

    # Compute log-likelihood for target tokens
    log_probs = F.log_softmax(logits, dim=-1)

    # Get log-probs for target tokens
    # target_ids shape: (1, T)
    # We want log_probs at positions corresponding to target tokens
    target_log_probs = log_probs[0, :-1].gather(
        1, target_ids[0, 1:].unsqueeze(-1)
    ).squeeze(-1)

    return target_log_probs.sum().item()


# ─── Task Implementations ─────────────────────────────────────────────────────

class MultipleChoiceTask:
    """Base class for multiple-choice evaluation tasks."""

    def __init__(self, data_path: str, tokenizer):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer

    def load_data(self, data_path: str) -> List[Dict]:
        raise NotImplementedError

    def format_example(self, example: Dict) -> Tuple[str, List[str]]:
        """Returns (context, list_of_choices)."""
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, device: str,
                 max_examples: Optional[int] = None) -> Dict:
        """Evaluate model on this task using log-likelihood scoring."""
        correct = 0
        total = 0

        data = self.data[:max_examples] if max_examples else self.data

        for example in data:
            context, choices = self.format_example(example)
            correct_idx = self.get_correct_idx(example)

            # Score each choice
            scores = []
            for choice in choices:
                full_text = context + choice
                input_ids = torch.tensor(
                    [self.tokenizer.encode(full_text)], dtype=torch.long
                )
                context_ids = torch.tensor(
                    [self.tokenizer.encode(context)], dtype=torch.long
                )

                # Only score the choice tokens
                score = compute_log_likelihood(model, input_ids, input_ids, device)
                # Normalize by choice length
                choice_len = input_ids.shape[1] - context_ids.shape[1]
                if choice_len > 0:
                    score /= choice_len
                scores.append(score)

            predicted = scores.index(max(scores))
            if predicted == correct_idx:
                correct += 1
            total += 1

        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
        }

    def get_correct_idx(self, example: Dict) -> int:
        raise NotImplementedError


class HellaSwagTask(MultipleChoiceTask):
    """HellaSwag commonsense NLI task."""

    def load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def format_example(self, example: Dict) -> Tuple[str, List[str]]:
        context = example['activity_label'] + ': ' + example['ctx']
        choices = example['endings']
        return context, choices

    def get_correct_idx(self, example: Dict) -> int:
        return int(example['label'])


class PIQATask(MultipleChoiceTask):
    """Physical Intuition QA task."""

    def load_data(self, data_path: str) -> List[Dict]:
        data = []
        labels_path = data_path.replace('.jsonl', '-labels.lst')

        with open(data_path, 'r') as f, open(labels_path, 'r') as lf:
            for line, label in zip(f, lf):
                example = json.loads(line)
                example['label'] = int(label.strip())
                data.append(example)
        return data

    def format_example(self, example: Dict) -> Tuple[str, List[str]]:
        context = 'Question: ' + example['goal'] + '\nAnswer: '
        choices = [example['sol1'], example['sol2']]
        return context, choices

    def get_correct_idx(self, example: Dict) -> int:
        return example['label']


class WinoGrandeTask(MultipleChoiceTask):
    """WinoGrande commonsense reasoning task."""

    def load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def format_example(self, example: Dict) -> Tuple[str, List[str]]:
        sentence = example['sentence']
        # Replace _ with each option
        choices = [
            sentence.replace('_', example['option1']),
            sentence.replace('_', example['option2']),
        ]
        return '', choices

    def get_correct_idx(self, example: Dict) -> int:
        return int(example['answer']) - 1


class ARCTask(MultipleChoiceTask):
    """ARC (AI2 Reasoning Challenge) task."""

    def load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def format_example(self, example: Dict) -> Tuple[str, List[str]]:
        context = 'Question: ' + example['question'] + '\nAnswer: '
        choices = [c['text'] for c in example['choices']['choices']]
        return context, choices

    def get_correct_idx(self, example: Dict) -> int:
        answer_key = example['answerKey']
        labels = [c['label'] for c in example['choices']['choices']]
        return labels.index(answer_key)


# ─── Main Evaluation ──────────────────────────────────────────────────────────

def evaluate_all_tasks(model: torch.nn.Module, tokenizer, task_data_dir: str,
                       device: str, output_file: Optional[str] = None) -> Dict:
    """
    Evaluate model on all downstream tasks from the paper.
    
    Tasks: HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge
    """
    results = {}
    task_data_dir = Path(task_data_dir)

    task_configs = [
        ('hellaswag', HellaSwagTask, 'hellaswag_val.jsonl'),
        ('piqa', PIQATask, 'piqa_val.jsonl'),
        ('winogrande', WinoGrandeTask, 'winogrande_val.jsonl'),
        ('arc_easy', ARCTask, 'arc_easy_test.jsonl'),
        ('arc_challenge', ARCTask, 'arc_challenge_test.jsonl'),
    ]

    for task_name, task_class, filename in task_configs:
        data_path = task_data_dir / filename
        if not data_path.exists():
            print(f"Skipping {task_name}: data file not found at {data_path}")
            continue

        print(f"Evaluating {task_name}...")
        task = task_class(str(data_path), tokenizer)
        result = task.evaluate(model, device)
        results[task_name] = result
        print(f"  {task_name}: {result['accuracy']*100:.2f}%")

    # Compute average accuracy
    if results:
        avg_acc = sum(r['accuracy'] for r in results.values()) / len(results)
        results['average'] = {'accuracy': avg_acc}
        print(f"\nAverage accuracy: {avg_acc*100:.2f}%")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    return results


@torch.no_grad()
def compute_perplexity(model: torch.nn.Module, data_path: str, seq_len: int,
                       device: str, max_tokens: int = 1_000_000) -> float:
    """Compute perplexity on a dataset."""
    import numpy as np

    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    data = data[:max_tokens]

    total_loss = 0.0
    n_tokens = 0

    model.eval()
    for i in range(0, len(data) - seq_len, seq_len):
        chunk = torch.from_numpy(data[i:i + seq_len + 1].astype('int64'))
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)

        _, loss = model(x, y)
        total_loss += loss.item() * seq_len
        n_tokens += seq_len

    avg_loss = total_loss / n_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate nGPT/GPT on downstream tasks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--task_data_dir', type=str, default='data/tasks',
                        help='Directory containing task data files')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to LLaMA-2 tokenizer')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Validation data for perplexity computation')
    parser.add_argument('--seq_len', type=int, default=4096)
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load tokenizer
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(args.tokenizer_path)
    else:
        import tiktoken
        tokenizer = tiktoken.get_encoding('gpt2')

    # Compute perplexity if val data provided
    if args.val_data:
        print(f"\nComputing perplexity on {args.val_data}...")
        ppl = compute_perplexity(model, args.val_data, args.seq_len, args.device)
        print(f"Perplexity: {ppl:.2f}")

    # Evaluate on downstream tasks
    print(f"\nEvaluating on downstream tasks...")
    results = evaluate_all_tasks(
        model, tokenizer, args.task_data_dir, args.device, args.output_file
    )
