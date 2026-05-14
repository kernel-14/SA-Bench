import os
import json
from datasets import Dataset

def load_hh_rlhf_dataset(data_dir, split='train'):
    """
    Load and preprocess the Anthropic Helpful and Harmless (HH-RLHF) dataset.
    Args:
        data_dir (str): Directory where data is stored.
        split (str): Split of dataset to load ('train' or 'validation').
    Returns:
        A dictionary containing dialogue prompts and responses.
    """
    dataset_path = os.path.join(data_dir, f'hh_rlhf_{split}.json')

    with open(dataset_path, 'r') as file:
        raw_data = json.load(file)

    processed_data = {
        'prompts': [],
        'responses': []
    }

    for instance in raw_data:
        prompt = instance.get('prompt', '').strip()
        response = instance.get('response', '').strip()
        if prompt and response:
            processed_data['prompts'].append(prompt)
            processed_data['responses'].append(response)

    return processed_data

if __name__ == '__main__':
    data_directory = os.getenv('DATASET_DIR', 'datasets/hh_rlhf')
    validation_data = load_hh_rlhf_dataset(data_directory, split='validation')
    print(f'Loaded {len(validation_data['prompts'])} validation samples from HH-RLHF dataset.')
