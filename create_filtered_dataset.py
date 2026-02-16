"""
Create a filtered dataset with maximum diversity for RL training.

This script implements sophisticated diversity-based filtering to create
a high-quality subset of data for reinforcement learning, while the remaining
data is used for supervised fine-tuning.

Strategy:
1. Ensure coverage of all categories
2. Maximize unique concept coverage (source + target)
3. Maximize unique node coverage in knowledge graph paths
4. Ensure diverse path patterns
5. Prefer examples with rare nodes/concepts to maintain long-tail coverage

The filtered dataset is typically used for RL (GRPO) training, while the
remaining examples are used for SFT training.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Any
import random
from datasets import load_from_disk, Dataset, DatasetDict


def analyze_dataset(dataset) -> Dict[str, Any]:
    """
    Analyze dataset to understand diversity patterns.
    
    Args:
        dataset: HuggingFace Dataset with KG metadata
    
    Returns:
        Dictionary containing statistics about the dataset
    """
    categories = [example['category'] for example in dataset]
    source_concepts = [example['source_concept'] for example in dataset]
    target_concepts = [example['target_concept'] for example in dataset]
    
    # Count frequencies
    category_counts = Counter(categories)
    source_concept_counts = Counter(source_concepts)
    target_concept_counts = Counter(target_concepts)
    
    # Analyze knowledge graph paths
    all_nodes = set()
    node_frequency = defaultdict(int)
    path_patterns = []
    concept_pair_frequency = defaultdict(int)
    
    for example in dataset:
        # Path pattern (sequence of relations in KG)
        relations_sequence = tuple([path['relation'] for path in example['paths']])
        path_patterns.append(relations_sequence)
        
        # Node frequency in KG
        for path in example['paths']:
            all_nodes.add(path['start'])
            all_nodes.add(path['end'])
            node_frequency[path['start']] += 1
            node_frequency[path['end']] += 1
        
        # Concept pairs
        concept_pair = (example['source_concept'], example['target_concept'])
        concept_pair_frequency[concept_pair] += 1
    
    pattern_counts = Counter(path_patterns)
    
    return {
        'total_examples': len(dataset),
        'unique_categories': len(category_counts),
        'unique_source_concepts': len(source_concept_counts),
        'unique_target_concepts': len(target_concept_counts),
        'unique_concepts': len(set(source_concepts + target_concepts)),
        'unique_nodes': len(all_nodes),
        'unique_path_patterns': len(set(path_patterns)),
        'unique_concept_pairs': len(concept_pair_frequency),
        'category_counts': category_counts,
        'source_concept_counts': source_concept_counts,
        'target_concept_counts': target_concept_counts,
        'node_frequency': node_frequency,
        'pattern_counts': pattern_counts,
        'concept_pair_frequency': concept_pair_frequency
    }


def compute_example_diversity_score(example, stats: Dict[str, Any]) -> float:
    """
    Compute diversity score for an example based on rarity of its components.
    
    Higher score = more diverse/rare = higher priority for inclusion in filtered set.
    
    This encourages the filtered dataset to maintain coverage of rare examples,
    which is important for maintaining long-tail performance.
    
    Args:
        example: Single dataset example
        stats: Statistics from analyze_dataset()
    
    Returns:
        Diversity score (higher = more diverse)
    """
    score = 0.0
    
    # Category rarity (inverse frequency)
    category_freq = stats['category_counts'][example['category']]
    score += 1.0 / (category_freq + 1)  # +1 for smoothing
    
    # Source concept rarity
    source_freq = stats['source_concept_counts'][example['source_concept']]
    score += 1.0 / (source_freq + 1)
    
    # Target concept rarity
    target_freq = stats['target_concept_counts'][example['target_concept']]
    score += 1.0 / (target_freq + 1)
    
    # Concept pair rarity (weighted higher)
    concept_pair = (example['source_concept'], example['target_concept'])
    pair_freq = stats['concept_pair_frequency'][concept_pair]
    score += 2.0 / (pair_freq + 1)
    
    # Path pattern rarity
    relations_sequence = tuple([path['relation'] for path in example['paths']])
    pattern_freq = stats['pattern_counts'][relations_sequence]
    score += 1.0 / (pattern_freq + 1)
    
    # Node rarity in KG paths
    unique_nodes_in_example = set()
    for path in example['paths']:
        unique_nodes_in_example.add(path['start'])
        unique_nodes_in_example.add(path['end'])
    
    node_rarity_sum = sum(
        1.0 / (stats['node_frequency'][node] + 1) 
        for node in unique_nodes_in_example
    )
    score += node_rarity_sum / len(unique_nodes_in_example) if unique_nodes_in_example else 0
    
    return score


def stratified_sampling_filter(
    dataset,
    target_size: int,
    min_per_category: int = 1
) -> List[int]:
    """
    Stratified sampling that ensures maximum diversity.
    
    Strategy:
    1. All categories are represented (at least min_per_category examples each)
    2. Remaining slots filled by diversity score
    3. Maximum coverage of concepts, nodes, and path patterns
    
    Args:
        dataset: HuggingFace Dataset
        target_size: Target number of examples in filtered dataset
        min_per_category: Minimum examples per category
    
    Returns:
        List of selected indices
    """
    print("Analyzing dataset for filtering...")
    stats = analyze_dataset(dataset)
    
    print(f"Dataset analysis:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Unique categories: {stats['unique_categories']}")
    print(f"  Unique concepts: {stats['unique_concepts']}")
    print(f"  Unique nodes: {stats['unique_nodes']}")
    print(f"  Unique path patterns: {stats['unique_path_patterns']}")
    
    # Group examples by category
    category_examples = defaultdict(list)
    for idx, example in enumerate(dataset):
        category_examples[example['category']].append(idx)
    
    selected_indices = []
    
    # Step 1: Ensure minimum representation per category
    for category, indices in category_examples.items():
        if len(indices) <= min_per_category:
            # Include all examples if category has few examples
            selected_indices.extend(indices)
        else:
            # Sample min_per_category examples with highest diversity scores
            example_scores = [
                (compute_example_diversity_score(dataset[idx], stats), idx)
                for idx in indices
            ]
            example_scores.sort(reverse=True)  # Highest score first
            selected_indices.extend([idx for _, idx in example_scores[:min_per_category]])
    
    print(f"Selected {len(selected_indices)} examples for minimum category coverage")
    
    # Step 2: Fill remaining slots with highest diversity examples
    remaining_slots = target_size - len(selected_indices)
    if remaining_slots > 0:
        # Get all unused examples and their diversity scores
        unused_indices = set(range(len(dataset))) - set(selected_indices)
        unused_scores = [
            (compute_example_diversity_score(dataset[idx], stats), idx)
            for idx in unused_indices
        ]
        
        # Sort by diversity score and take top remaining_slots
        unused_scores.sort(reverse=True)
        additional_indices = [idx for _, idx in unused_scores[:remaining_slots]]
        selected_indices.extend(additional_indices)
        
        print(f"Added {len(additional_indices)} examples based on diversity scores")
    
    return selected_indices


def create_filtered_dataset(
    input_path: str,
    output_path: str,
    target_size: int,
    min_per_category: int = 2,
    save_metadata: bool = True
):
    """
    Create filtered dataset with maximum diversity for RL training.
    
    The filtered dataset typically contains 5-10k examples selected for maximum
    diversity. This filtered set is used for RL training, while the remaining
    examples are used for SFT training.
    
    Args:
        input_path: Path to input tokenized dataset
        output_path: Path to save filtered dataset
        target_size: Target number of examples in filtered dataset
        min_per_category: Minimum examples per category
        save_metadata: Whether to save filtering metadata
    """
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    train_dataset = dataset['train']
    
    print(f"Original dataset size: {len(train_dataset)}")
    print(f"Target filtered size: {target_size}")
    
    # Get filtered indices using diversity-based stratified sampling
    selected_indices = stratified_sampling_filter(
        train_dataset,
        target_size,
        min_per_category
    )
    
    print(f"Selected {len(selected_indices)} examples for filtered dataset")
    
    # Create filtered dataset
    filtered_data = {
        key: [train_dataset[i][key] for i in selected_indices]
        for key in train_dataset.column_names
    }
    
    filtered_dataset = Dataset.from_dict(filtered_data)
    filtered_dataset_dict = DatasetDict({'train': filtered_dataset})
    
    # Analyze filtered dataset
    print("\nAnalyzing filtered dataset...")
    filtered_stats = analyze_dataset(filtered_dataset)
    original_stats = analyze_dataset(train_dataset)
    
    print(f"Filtered dataset analysis:")
    print(f"  Total examples: {filtered_stats['total_examples']}")
    print(f"  Unique categories: {filtered_stats['unique_categories']}/{original_stats['unique_categories']}")
    print(f"  Unique concepts: {filtered_stats['unique_concepts']}/{original_stats['unique_concepts']}")
    print(f"  Unique nodes: {filtered_stats['unique_nodes']}/{original_stats['unique_nodes']}")
    print(f"  Unique path patterns: {filtered_stats['unique_path_patterns']}/{original_stats['unique_path_patterns']}")
    
    # Category distribution in filtered dataset
    print(f"\nCategory distribution (top 10):")
    for category, count in filtered_stats['category_counts'].most_common(10):
        original_count = original_stats['category_counts'][category]
        print(f"  {category}: {count}/{original_count} ({100*count/original_count:.1f}%)")
    
    # Save filtered dataset
    print(f"\nSaving filtered dataset to {output_path}...")
    filtered_dataset_dict.save_to_disk(output_path)
    
    # Save metadata
    if save_metadata:
        metadata = {
            'original_size': len(train_dataset),
            'filtered_size': len(filtered_dataset),
            'target_size': target_size,
            'min_per_category': min_per_category,
            'selected_indices': selected_indices,
            'coverage_stats': {
                'categories': f"{filtered_stats['unique_categories']}/{original_stats['unique_categories']}",
                'concepts': f"{filtered_stats['unique_concepts']}/{original_stats['unique_concepts']}",
                'nodes': f"{filtered_stats['unique_nodes']}/{original_stats['unique_nodes']}",
                'path_patterns': f"{filtered_stats['unique_path_patterns']}/{original_stats['unique_path_patterns']}"
            }
        }
        
        metadata_path = Path(output_path).parent / f"{Path(output_path).name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
    
    print("Filtering complete!")


def main():
    """Command-line interface for dataset filtering."""
    parser = argparse.ArgumentParser(
        description="Create filtered dataset with maximum diversity for RL training"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input tokenized dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save filtered dataset"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=5000,
        help="Target size for filtered dataset (default: 5000)"
    )
    parser.add_argument(
        "--min_per_category",
        type=int,
        default=2,
        help="Minimum examples per category (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    create_filtered_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        target_size=args.target_size,
        min_per_category=args.min_per_category
    )


if __name__ == "__main__":
    main()
