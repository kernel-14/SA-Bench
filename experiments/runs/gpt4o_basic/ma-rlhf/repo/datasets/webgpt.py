import os
import json
from datasets import load_dataset

def load_webgpt_dataset(data_dir, split='train'):
    """
    Load and preprocess the WebGPT Comparison dataset for question answering tasks.
    Args:
        data_dir (str): Directory where data is stored.
        split (str): Split of dataset to load ('train' or 'validation').
    Returns:
        A dictionary containing questions and answers.
    """
    dataset_path = os.path.join(data_dir, f'webgpt_comparison_{split}.json')

    with open(dataset_path, 'r') as file:
        raw_data = json.load(file)

    processed_data = {
        'questions': [],
        'answers': []
    }

    for instance in raw_data:
        question = instance.get('question', '').strip()
        answer = instance.get('answer', '').strip()
        if question and answer:
            processed_data['questions'].append(question)
            processed_data['answers'].append(answer)

    return processed_data

if __name__ == '__main__':
    data_directory = os.getenv('DATASET_DIR', 'datasets/webgpt')
    validation_data = load_webgpt_dataset(data_directory, split='validation')
    print(f'Loaded {len(validation_data['questions'])} validation samples from WebGPT Comparison dataset.')
