"""
GPT-4 pairwise evaluation script.

Evaluates win rates between MA-PPO and vanilla PPO using GPT-4 as judge.
Implements the evaluation prompts from Appendix F.1 of the paper.

Usage:
    python gpt4_eval.py \
        --task tldr \
        --model_a_responses responses_ma_ppo.jsonl \
        --model_b_responses responses_ppo.jsonl \
        --output results.json
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# GPT-4 evaluation prompts from Appendix F.1
TLDR_EVAL_PROMPT = """You will be given two summaries written for an article. Your task is to pick the better one between them, based on the four criteria. Please make sure you read and understand these instructions carefully. Relevance - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Coherence - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Consistency - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.

Fluency - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

You should output single character to indicate which summary you think is better. 'A' stands for Summary A and 'B' stands for Summary B. If you think both summaries are equally good, output 'E'

Article / Post:{article}

Summary A:{summary_a}

Summary B:{summary_b}

Your Choice (only a single character):"""


HH_RLHF_EVAL_PROMPT = """For the following query to a chatbot assistant, which response is more helpful?

First provide a one-sentence comparison of the two responses and explain which you feel is more helpful. Second, on a new line, state only 'A' or 'B' to indicate which response is more helpful. If they are equally good or bad, state 'E'. Your response should use the json format, with "comparison" and "choice" as keys.

Query: {query}
Response A: {response_a}
Response B: {response_b}
Your Judgment:"""


WEBGPT_EVAL_PROMPT = """You will be given two response written for an question. Your task is to pick the better one between them, based on these criteria.

Factual accuracy - which answer is more factually accurate?

Coherence - which answer is easier to follow?

Usefulness overall - all things considered, which answer would be more helpful to the person who asked this question?

You should output with a json format where the key is the criteria and the value is the choice you made, using 'A' stands for Response A and 'B' stands for Response B. If you think both responses are equally good, output 'E'.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Your Judgment (you should also output the reason, note that you are allowed to think both responses are equally good, then output with 'E'):"""


def get_eval_prompt(task: str, article: str, response_a: str, response_b: str) -> str:
    """Build the GPT-4 evaluation prompt for the given task."""
    if task == "tldr":
        return TLDR_EVAL_PROMPT.format(
            article=article,
            summary_a=response_a,
            summary_b=response_b,
        )
    elif task == "hh_rlhf":
        return HH_RLHF_EVAL_PROMPT.format(
            query=article,
            response_a=response_a,
            response_b=response_b,
        )
    elif task == "webgpt":
        return WEBGPT_EVAL_PROMPT.format(
            question=article,
            answer_a=response_a,
            answer_b=response_b,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def parse_gpt4_response(task: str, response: str) -> str:
    """
    Parse GPT-4 response to extract the choice ('A', 'B', or 'E').

    Returns:
        'A', 'B', or 'E' (tie).
    """
    response = response.strip()

    if task == "tldr":
        # Single character response
        for char in response:
            if char in ("A", "B", "E"):
                return char
        return "E"

    elif task in ("hh_rlhf", "webgpt"):
        # JSON format
        try:
            data = json.loads(response)
            if task == "hh_rlhf":
                choice = data.get("choice", "E")
            else:
                # WebGPT: majority vote across criteria
                choices = list(data.values())
                a_count = choices.count("A")
                b_count = choices.count("B")
                if a_count > b_count:
                    choice = "A"
                elif b_count > a_count:
                    choice = "B"
                else:
                    choice = "E"
            return choice if choice in ("A", "B", "E") else "E"
        except json.JSONDecodeError:
            # Fallback: look for A/B/E in text
            for char in response:
                if char in ("A", "B", "E"):
                    return char
            return "E"

    return "E"


def evaluate_with_gpt4(
    task: str,
    articles: List[str],
    responses_a: List[str],
    responses_b: List[str],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-2024-05-13",
    randomize_order: bool = True,
) -> Dict:
    """
    Evaluate win rates using GPT-4 as judge.

    Following the paper (Section 4.1), we randomize the order of responses
    to mitigate potential evaluation biases.

    Args:
        task: Task name ('tldr', 'hh_rlhf', 'webgpt').
        articles: List of source articles/prompts.
        responses_a: Responses from model A (MA-PPO).
        responses_b: Responses from model B (vanilla PPO).
        api_key: OpenAI API key.
        model: GPT-4 model to use (default: gpt-4o-2024-05-13).
        randomize_order: Whether to randomize A/B order.

    Returns:
        Dict with win_rate_a, win_rate_b, tie_rate, and per-sample results.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    results = []
    wins_a = 0
    wins_b = 0
    ties = 0

    for i, (article, resp_a, resp_b) in enumerate(zip(articles, responses_a, responses_b)):
        # Randomize order to mitigate position bias
        if randomize_order and random.random() < 0.5:
            first, second = resp_a, resp_b
            swapped = False
        else:
            first, second = resp_b, resp_a
            swapped = True

        prompt = get_eval_prompt(task, article, first, second)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            gpt4_output = response.choices[0].message.content
            choice = parse_gpt4_response(task, gpt4_output)

            # Adjust for swapped order
            if swapped:
                if choice == "A":
                    choice = "B"
                elif choice == "B":
                    choice = "A"

            if choice == "A":
                wins_a += 1
            elif choice == "B":
                wins_b += 1
            else:
                ties += 1

            results.append({
                "idx": i,
                "choice": choice,
                "gpt4_output": gpt4_output,
            })

        except Exception as e:
            logger.warning(f"GPT-4 API error at sample {i}: {e}")
            ties += 1
            results.append({"idx": i, "choice": "E", "error": str(e)})

    n = len(articles)
    return {
        "win_rate_a": wins_a / n if n > 0 else 0.0,
        "win_rate_b": wins_b / n if n > 0 else 0.0,
        "tie_rate": ties / n if n > 0 else 0.0,
        "n_samples": n,
        "per_sample": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["tldr", "hh_rlhf", "webgpt"])
    parser.add_argument("--articles", required=True, help="JSONL file with source articles")
    parser.add_argument("--model_a_responses", required=True, help="JSONL with MA-PPO responses")
    parser.add_argument("--model_b_responses", required=True, help="JSONL with PPO responses")
    parser.add_argument("--output", default="gpt4_eval_results.json")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default="gpt-4o-2024-05-13")
    args = parser.parse_args()

    def load_jsonl(path):
        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    articles_data = load_jsonl(args.articles)
    responses_a = load_jsonl(args.model_a_responses)
    responses_b = load_jsonl(args.model_b_responses)

    # Sample n_samples instances
    n = min(args.n_samples, len(articles_data))
    indices = random.sample(range(len(articles_data)), n)

    articles = [articles_data[i].get("article", articles_data[i].get("prompt", "")) for i in indices]
    resp_a = [responses_a[i].get("response", "") for i in indices]
    resp_b = [responses_b[i].get("response", "") for i in indices]

    results = evaluate_with_gpt4(
        task=args.task,
        articles=articles,
        responses_a=resp_a,
        responses_b=resp_b,
        api_key=args.api_key,
        model=args.model,
    )

    logger.info(f"Win rate (MA-PPO): {results['win_rate_a']:.2%}")
    logger.info(f"Win rate (PPO): {results['win_rate_b']:.2%}")
    logger.info(f"Tie rate: {results['tie_rate']:.2%}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
