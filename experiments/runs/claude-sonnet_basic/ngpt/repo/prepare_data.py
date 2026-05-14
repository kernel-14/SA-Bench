"""
Data preparation script for OpenWebText dataset.

Downloads and tokenizes the OpenWebText dataset using the LLaMA-2 tokenizer,
saving the result as memory-mapped binary files for efficient training.

Usage:
    python prepare_data.py --output_dir data/ --tokenizer_path /path/to/tokenizer.model

The paper uses:
- OpenWebText dataset (Gokaslan & Cohen, 2019)
- LLaMA-2 tokenizer with 32k tokens
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def prepare_openwebtext(output_dir: str, tokenizer_path: str = None,
                        val_fraction: float = 0.005, num_proc: int = 8):
    """
    Download and tokenize OpenWebText dataset.
    
    Args:
        output_dir: Directory to save tokenized data
        tokenizer_path: Path to LLaMA-2 tokenizer model file
        val_fraction: Fraction of data to use for validation
        num_proc: Number of processes for tokenization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        # Use sentencepiece tokenizer (LLaMA-2)
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        vocab_size = sp.GetPieceSize()
        print(f"Loaded LLaMA-2 tokenizer with vocab size: {vocab_size}")

        def tokenize(text):
            return sp.Encode(text, out_type=int)
    else:
        # Fallback: use GPT-2 tokenizer from tiktoken
        print("LLaMA-2 tokenizer not found, falling back to GPT-2 tokenizer")
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        vocab_size = enc.n_vocab

        def tokenize(text):
            return enc.encode_ordinary(text)

    # Load dataset
    try:
        from datasets import load_dataset
        print("Loading OpenWebText dataset...")
        dataset = load_dataset('openwebtext', num_proc=num_proc)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please install: pip install datasets")
        return

    # Split into train/val
    split_dataset = dataset['train'].train_test_split(
        test_size=val_fraction, seed=42, shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    print(f"Train size: {len(split_dataset['train'])}")
    print(f"Val size: {len(split_dataset['val'])}")

    # Tokenize
    def process(example):
        ids = tokenize(example['text'])
        ids.append(2)  # EOS token (</s> in LLaMA-2)
        return {'ids': ids, 'len': len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc='Tokenizing',
        num_proc=num_proc,
    )

    # Save as binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = output_dir / f'{split}.bin'
        dtype = np.uint16  # vocab size < 65536

        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"Saved {split}.bin: {arr_len:,} tokens")

    print(f"\nData preparation complete. Files saved to {output_dir}")
    print(f"Vocab size: {vocab_size}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare OpenWebText dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for tokenized data')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to LLaMA-2 tokenizer.model file')
    parser.add_argument('--val_fraction', type=float, default=0.005,
                        help='Fraction of data for validation')
    parser.add_argument('--num_proc', type=int, default=8,
                        help='Number of processes for tokenization')
    args = parser.parse_args()

    prepare_openwebtext(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        val_fraction=args.val_fraction,
        num_proc=args.num_proc,
    )
