"""
Evaluation script for NaViL on multimodal benchmarks.

Benchmarks from the paper:
MLLM benchmarks (Table 1):
  - MMVet
  - MMMU val
  - MMBench-EN test
  - MME
  - MathVista MINI
  - OCRBench
  - CCBench

VQA benchmarks (Table 2):
  - TextVQA val
  - ScienceQA-IMG test
  - GQA test dev
  - DocVQA test
  - AI2D test
  - ChartQA test
  - InfographicVQA test
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from navil.model import NaViLModel, NaViLConfig
from navil.data import ImageProcessor

logger = logging.getLogger(__name__)


class NaViLEvaluator:
    """
    Evaluator for NaViL on multimodal benchmarks.
    """

    def __init__(
        self,
        model: NaViLModel,
        tokenizer,
        image_processor: ImageProcessor,
        device: str = "cuda",
        max_new_tokens: int = 512,
        use_multiscale: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_multiscale = use_multiscale

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Generate response for an image-text prompt.

        Args:
            image: PIL Image or None for text-only
            prompt: text prompt
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling parameter
        Returns:
            generated text
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        # Process image
        image_tensor = None
        if image is not None:
            image_tensor = self.image_processor.process(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_ids = self._generate_tokens(
                input_ids=input_ids,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # Decode
        new_tokens = generated_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Auto-regressive token generation."""
        B = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None

        # Encode image if provided
        visual_tokens = None
        if images is not None:
            visual_tokens, _ = self.model.visual_encoder(images)

        for step in range(max_new_tokens):
            # Forward pass
            if step == 0:
                # First step: process full sequence
                outputs = self.model(
                    input_ids=generated,
                    images=images,
                    use_multiscale=self.use_multiscale,
                )
            else:
                # Subsequent steps: only process new token
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                )

            past_key_values = outputs["past_key_values"]
            logits = outputs["logits"][:, -1, :]  # (B, vocab_size)

            # Sample next token
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_p < 1.0:
                    logits = self._top_p_filter(logits, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return generated

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")

        # Scatter back
        logits = torch.scatter(logits, 1, sorted_indices, sorted_logits)
        return logits

    def evaluate_vqa(
        self,
        dataset: List[Dict],
        benchmark_name: str,
    ) -> Dict:
        """
        Evaluate on a VQA benchmark.

        Args:
            dataset: list of {"image": path, "question": str, "answer": str}
            benchmark_name: name of the benchmark
        Returns:
            dict with accuracy and predictions
        """
        correct = 0
        total = 0
        predictions = []

        for item in dataset:
            image = None
            if "image" in item and item["image"]:
                try:
                    image = Image.open(item["image"])
                except Exception:
                    pass

            question = item.get("question", "")
            gt_answer = item.get("answer", "")

            # Format prompt
            prompt = self._format_vqa_prompt(question, benchmark_name)

            # Generate answer
            pred = self.generate(image, prompt)

            # Check correctness
            is_correct = self._check_answer(pred, gt_answer, benchmark_name)
            correct += int(is_correct)
            total += 1

            predictions.append({
                "question": question,
                "gt_answer": gt_answer,
                "prediction": pred,
                "correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"{benchmark_name}: {accuracy:.4f} ({correct}/{total})")

        return {
            "benchmark": benchmark_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions,
        }

    def _format_vqa_prompt(self, question: str, benchmark: str) -> str:
        """Format prompt for different benchmarks."""
        # InternLM2 conversation format
        system = "You are a helpful assistant."
        if benchmark in ["TextVQA", "DocVQA", "OCRBench"]:
            instruction = f"<image>\n{question}\nAnswer the question using a single word or phrase."
        elif benchmark in ["MMVet", "MMMU"]:
            instruction = f"<image>\n{question}"
        elif benchmark == "MathVista":
            instruction = f"<image>\n{question}\nAnswer:"
        else:
            instruction = f"<image>\n{question}\nAnswer:"

        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    def _check_answer(self, pred: str, gt: str, benchmark: str) -> bool:
        """Check if prediction matches ground truth."""
        pred = pred.strip().lower()
        gt = gt.strip().lower()

        if benchmark in ["GQA", "TextVQA", "ScienceQA"]:
            # Exact match or substring
            return pred == gt or gt in pred or pred in gt
        elif benchmark in ["DocVQA", "ChartQA", "InfoVQA"]:
            # ANLS-style (simplified)
            return pred == gt or gt in pred
        else:
            return pred == gt or gt in pred


def load_model(checkpoint_path: str, model_name: str = "NaViL-2B") -> NaViLModel:
    """Load NaViL model from checkpoint."""
    if model_name == "NaViL-2B":
        config = NaViLConfig.navil_2b()
    else:
        config = NaViLConfig.navil_9b()

    model = NaViLModel(config)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate NaViL")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="NaViL-2B")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["MMVet", "MMMU", "MMBench", "MME", "MathVista",
                                 "OCRBench", "CCBench", "TextVQA", "ScienceQA",
                                 "GQA", "DocVQA", "AI2D", "ChartQA", "InfoVQA"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./eval_results")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_multiscale", action="store_true", default=True)
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, args.model_name)
    model = model.to(device).eval()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "internlm/internlm2-1_8b", trust_remote_code=True
    )

    # Build evaluator
    image_processor = ImageProcessor()
    evaluator = NaViLEvaluator(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        max_new_tokens=args.max_new_tokens,
        use_multiscale=args.use_multiscale,
    )

    # Load dataset
    with open(args.data_path) as f:
        dataset = json.load(f)

    # Evaluate
    results = evaluator.evaluate_vqa(dataset, args.benchmark)

    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"{args.benchmark}_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{args.benchmark} Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Correct: {results['correct']}/{results['total']}")
    print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
