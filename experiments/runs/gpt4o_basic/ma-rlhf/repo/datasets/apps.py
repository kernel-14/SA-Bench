import os
import json

def load_apps_dataset(data_dir, split='train'):
    """
    Load and preprocess the APPS dataset for code generation tasks.
    Args:
        data_dir (str): Directory where data is stored.
        split (str): Split of dataset to load ('train' or 'validation').
    Returns:
        A dictionary containing problem descriptions and solutions.
    """
    dataset_path = os.path.join(data_dir, f'apps_{split}.json')

    with open(dataset_path, 'r') as file:
        raw_data = json.load(file)

    processed_data = {
        'descriptions': [],
        'solutions': []
    }

    for instance in raw_data:
        description = instance.get('description', '').strip()
        solution = instance.get('solution', '').strip()
        if description and solution:
            processed_data['descriptions'].append(description)
            processed_data['solutions'].append(solution)

    return processed_data

if __name__ == '__main__':
    data_directory = os.getenv('DATASET_DIR', 'datasets/apps')
    train_data = load_apps_dataset(data_directory, split='train')
    print(f'Loaded {len(train_data['descriptions'])} training samples from APPS dataset.')
