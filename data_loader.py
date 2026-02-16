"""
Data loading utilities for Knowledge Graph-guided RL training.

This script provides utilities to load and prepare training data.
The actual training data is not included in this repository.
"""

import os
from typing import Optional
from datasets import load_from_disk, Dataset, DatasetDict


def load_training_data(
    dataset_path: str,
    split: str = "train"
) -> Dataset:
    """
    Load training dataset from disk.
    
    Args:
        dataset_path: Path to the tokenized dataset directory
        split: Dataset split to load ('train', 'test', or 'validation')
    
    Returns:
        Dataset object containing the training data
    
    Expected dataset format:
        - 'text': Tokenized conversation in chat format
        - 'question_and_explanation': Original question with explanation
        - 'paths': Knowledge graph paths (optional, for RL training)
        - 'answer': Ground truth answer letter (A/B/C/D)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please prepare your dataset using the data_prep.py script."
        )
    
    dataset = load_from_disk(dataset_path)
    
    # Handle DatasetDict vs Dataset
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(
                f"Split '{split}' not found. Available splits: {available_splits}"
            )
        dataset = dataset[split]
    
    return dataset


def load_sft_dataset(
    full_dataset_path: str,
    filtered_dataset_path: Optional[str] = None,
    use_full_data: bool = True
) -> DatasetDict:
    """
    Load dataset for SFT training.
    
    Args:
        full_dataset_path: Path to the full tokenized dataset
        filtered_dataset_path: Path to filtered examples (optional)
        use_full_data: If True, use all data. If False, exclude filtered examples.
    
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    full = load_from_disk(full_dataset_path)
    full_train = full['train'] if 'train' in full else full
    
    if use_full_data or filtered_dataset_path is None:
        # Use all data
        if 'test' in full:
            eval_ds = full['test']
        else:
            # Create a small eval split
            holdout = min(500, max(1, int(0.02 * len(full_train))))
            eval_ds = full_train.select(range(0, holdout))
            full_train = full_train.select(range(holdout, len(full_train)))
        
        return DatasetDict({'train': full_train, 'test': eval_ds})
    
    # Exclude filtered examples (for complementary train/RL splits)
    filtered = load_from_disk(filtered_dataset_path)
    filtered_train = filtered['train'] if 'train' in filtered else filtered
    
    filtered_texts = set(filtered_train['text'])
    keep_indices = [i for i, t in enumerate(full_train['text']) if t not in filtered_texts]
    sft_train = full_train.select(keep_indices)
    
    if 'test' in full:
        eval_ds = full['test']
    else:
        holdout = min(500, max(1, int(0.02 * len(sft_train))))
        eval_ds = sft_train.select(range(0, holdout))
        sft_train = sft_train.select(range(holdout, len(sft_train)))
    
    print(f"SFT dataset - Train: {len(sft_train)}, Test: {len(eval_ds)}")
    return DatasetDict({'train': sft_train, 'test': eval_ds})


def load_rl_dataset(
    dataset_path: str,
    split: str = "train"
) -> Dataset:
    """
    Load dataset for RL training (GRPO).
    
    This typically uses a filtered/curated subset for RL fine-tuning.
    
    Args:
        dataset_path: Path to the RL training dataset
        split: Dataset split to load
    
    Returns:
        Dataset for RL training
    """
    return load_training_data(dataset_path, split=split)


# Placeholder configuration for dataset paths
# Replace these with your actual data paths
DATASET_PATHS = {
    "full_dataset": "/path/to/your/full_tokenized_dataset",
    "filtered_dataset": "/path/to/your/filtered_dataset",
    "rl_dataset": "/path/to/your/rl_dataset",
}


if __name__ == "__main__":
    # Example usage
    print("Data loader utilities for KG-guided RL training")
    print("\nTo use these functions, update the DATASET_PATHS configuration")
    print("with your actual dataset locations.")
    
    # Example: Load SFT dataset
    # dataset = load_sft_dataset(
    #     full_dataset_path=DATASET_PATHS["full_dataset"],
    #     use_full_data=True
    # )
    # print(f"Loaded {len(dataset['train'])} training examples")
