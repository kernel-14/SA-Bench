from typing import List, Dict
import json

class SCoReDataset:
    """
    A class to handle loading and preparing datasets for SCoRe training and evaluation.
    """

    def __init__(self, dataset_name: str, data_dir: str):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.problems: List[Dict[str, str]] = []
        self.ground_truths: List[str] = []
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the specified dataset. For this static reproduction, we will use dummy data.
        In a real scenario, this would load actual MATH or HumanEval data.
        """
        print(f"[DEBUG] Loading dataset: {self.dataset_name} from {self.data_dir}")

        if self.dataset_name == "MATH500":
            # Dummy data for MATH500
            self.problems = [
                {"problem": "If $n \equiv 2$ (mod 7), then find the remainder when $( n + 2 ) ( n + 4 ) ( n + 6 )$ is divided by 7."},
                {"problem": "Let $f ( x ) = \left\lfloor \left( - { \frac { 5 } { 8 } } ight) ^ { x } ightfloor$ be a function that is defined for all values of $x$ in $[ 0 , \infty )$ such that $f ( x )$ is a real number. How many distinct values exist in the range of $f ( x ) ?"},
                {"problem": "What is 2 + 2?"},
                {"problem": "Solve for x: 2x + 5 = 11"},
            ]
            self.ground_truths = [
                "3", # Correct answer for MATH Example 1
                "3", # Correct answer for MATH Example 2
                "4",
                "3",
            ]
        elif self.dataset_name == "HumanEval":
            # Dummy data for HumanEval
            self.problems = [
                {"problem": "Write a Python function to compute the Fibonacci sequence up to n."},
                {"problem": "Write a function that reverses a string."},
            ]
            self.ground_truths = [
                "def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        print(a, end=" ")
        a, b = b, a + b",
                "def reverse_string(s):
    return s[::-1]",
            ]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        print(f"Loaded {len(self.problems)} problems for {self.dataset_name}.")

    def get_data(self):
        return self.problems, self.ground_truths
