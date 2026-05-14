import os
import json
from datasets import load_dataset

def load_tldr_dataset(data_dir, split='train'):
    """
    Load and preprocess the TL;DR dataset for summarization.
    Args:
        data_dir (str): Directory where data is stored.
        split (str): Split of dataset to load ('train', 'validation', or 'test').
    Returns:
        A dictionary containing preprocessed prompts and summaries.
    """
    dataset = load_dataset('reddit_tldr', split=split)

    processed_data = {
        'prompts': [],
        'summaries': []
    }

    for instance in dataset:
        post = instance.get('content', '').strip()
        summary = instance.get('summary', '').strip()
        if post and summary:
            processed_data['prompts'].append(post)
            processed_data['summaries'].append(summary)

    return processed_data

if __name__ == '__main__':
    data_directory = os.getenv('DATASET_DIR', 'datasets/tldr')
    train_data = load_tldr_dataset(data_directory, split='train')
    print(f'Loaded {len(train_data['prompts'])} training samples from TL;DR dataset.')
