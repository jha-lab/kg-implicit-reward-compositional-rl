"""
Dataset preprocessing and splitting for SFT and RL training.

This script handles:
1. Converting raw data to messages format
2. Creating train/test splits
3. Preprocessing for GRPO training (with KG path integration)
"""

import os
import re
from typing import List, Dict, Optional, Any
from datasets import Dataset, DatasetDict, load_from_disk


def extract_answer(text: str) -> str:
    """
    Extract answer from text (supports multiple formats).
    
    Handles:
    - 'Final Answer: X' format
    - '<answer>X</answer>' tags
    - Markdown formatting (e.g., **Final Answer:** C)
    
    Args:
        text: Text containing the answer
    
    Returns:
        Extracted answer letter (A/B/C/D) or empty string
    """
    try:
        # Strip markdown formatting
        text_clean = re.sub(r'\*+', '', text)
        
        # Look for "Final Answer: X" pattern
        match = re.search(r'Final Answer\s*[:\-]\s*([A-D])', text_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fallback 1: Check for <answer>X</answer> tags
        answer_match = re.search(r'<answer>\s*([A-D])\s*</answer>', text_clean, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Fallback 2: Check for just <answer>X (no closing tag)
        answer_match2 = re.search(r'<answer>\s*([A-D])', text_clean, re.IGNORECASE)
        if answer_match2:
            return answer_match2.group(1).upper()
        
        # Fallback 3: Extract any A-D letter that appears after </think>
        if '</think>' in text_clean:
            after_think = text_clean.split('</think>')[-1]
            letters = re.findall(r'\b[A-D]\b', after_think)
            if letters:
                return letters[0].upper()
        
        return ""
    except Exception:
        return ""


def to_messages_format(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dataset example to messages format for TRL training.
    
    Expected input format (in 'question_and_explanation' field):
        <Question>: {question text}
        <Options>: {options A-D}
        <Explanation>: {chain of thought}
        <Answer>: {answer letter}
    
    Output format:
        {
            "messages": [
                {"role": "user", "content": "{question}\nOptions:{options}"},
                {"role": "assistant", "content": "<think>\n{explanation}\n</think>\nFinal Answer: {answer}"}
            ]
        }
    
    Args:
        example: Dataset example with 'question_and_explanation' field
    
    Returns:
        Dict with 'messages' field in chat format
    """
    qae = example.get('question_and_explanation', '')
    try:
        # Extract question
        if '<Question>:' in qae:
            question = qae.split('<Question>:')[1].split('</Question>')[0]
        elif '<Question>' in qae:
            question = qae.split('<Question>')[1].split('</Question>')[0]
        else:
            question = ''

        # Extract options
        if '<Options>' in qae:
            options = qae.split('<Options>')[1].split('</Options>')[0]
        elif '<Options>:' in qae:
            options = qae.split('<Options>:')[1].split('</Options>')[0]
        else:
            options = ''

        # Extract explanation (chain of thought)
        cot = qae.split('<Explanation>')[1].split('</Explanation>')[0]
        
        # Extract answer
        answer = qae.split('<Answer>:')[1].split('</Answer>')[0]

        # Create user prompt (question + options)
        user_content = question.strip() + '\nOptions:' + options.strip()
        
        # Create assistant response (thinking + final answer)
        assistant_content = "<think>\n" + cot.strip() + "\n</think>\nFinal Answer: " + answer.strip()
        
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
    except Exception as e:
        # Fallback to text field if parsing fails
        fallback_text = example.get('text', '')
        return {
            "messages": [
                {"role": "user", "content": "Answer the following question:"},
                {"role": "assistant", "content": fallback_text}
            ]
        }


def prepare_sft_dataset(
    dataset: Dataset,
    eval_split_ratio: float = 0.02,
    min_eval_size: int = 500
) -> DatasetDict:
    """
    Prepare dataset for SFT training.
    
    Args:
        dataset: Input dataset with raw examples
        eval_split_ratio: Ratio of data to use for evaluation
        min_eval_size: Minimum number of examples in eval split
    
    Returns:
        DatasetDict with 'train' and 'test' splits in messages format
    """
    # Convert to messages format
    dataset = dataset.map(to_messages_format, batched=False)
    
    # Create eval split if not present
    holdout = min(min_eval_size, max(1, int(eval_split_ratio * len(dataset))))
    eval_ds = dataset.select(range(0, holdout))
    train_ds = dataset.select(range(holdout, len(dataset)))
    
    print(f"SFT dataset prepared - Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return DatasetDict({'train': train_ds, 'test': eval_ds})


def preprocess_grpo_dataset(
    dataset_path: str,
    split: str = "train",
    chunk_size: int = 1000,
    enable_thinking: bool = True,
    system_prompt: str = "A conversation between user and assistant. The user asks a single-choice Multiple Choice Question, and the assistant solves it using step-by-step reasoning. Please answer the multiple choice question by selecting only one from option A, option B, option C, option D. \n\nThe assistant first thinks through the problem systematically, then provides the explanation and final answer. Use <think>...</think> tags for internal reasoning, then provide the explanation process and answer enclosed within <explanation> </explanation> and <answer> </answer> tags, respectively.",
    task_instructions: str = "Please provide complete and accurate answers with clear reasoning. The answer must only be a single letter from A, B, C, D."
) -> Dataset:
    """
    Preprocess dataset for GRPO (RL) training.
    
    Converts dataset to prompt/answer format with optional thinking mode.
    Includes knowledge graph path information if available.
    
    Args:
        dataset_path: Path to the input dataset
        split: Dataset split to load
        chunk_size: Batch size for processing
        enable_thinking: Whether to add thinking mode prompt
        system_prompt: System prompt for the model
        task_instructions: Task-specific instructions
    
    Returns:
        Processed dataset with 'prompt', 'answer', and 'paths' fields
    """
    dataset = load_from_disk(dataset_path)[split]
    
    def process_batch(batch):
        prompts = []
        answers = []
        paths_list = []
        
        has_paths = "paths" in batch
        for idx, text in enumerate(batch["text"]):
            # Parse conversation format
            # Expected: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
            parts = text.split("<|im_start|>assistant\n")
            if len(parts) >= 2:
                user_part = parts[0].replace("<|im_start|>user\n", "").replace("<|im_end|>", "").strip()
                assistant_part = parts[1].replace("<|im_end|>", "").strip()
                
                # Enhanced prompt for thinking mode
                prompt = [
                    {
                        "role": "system", 
                        "content": system_prompt + "\n" + task_instructions
                    },
                    {
                        "role": "user", 
                        "content": user_part + ("/think" if enable_thinking else "/no_think")
                    }
                ]
                
                prompts.append(prompt)
                answers.append(extract_answer(assistant_part))
                
                if has_paths:
                    paths_list.append(batch["paths"][idx])
        
        result = {
            "prompt": prompts,
            "answer": answers,
        }
        if has_paths:
            result["paths"] = paths_list
        
        return result
    
    return dataset.map(process_batch, batched=True, batch_size=chunk_size)




def main():
    """Command-line interface for data preprocessing."""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for SFT or RL training"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save processed dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sft", "rl"],
        required=True,
        help="Processing mode: 'sft' for supervised fine-tuning, 'rl' for reinforcement learning"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode for RL preprocessing (adds /think token)"
    )
    parser.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.02,
        help="Ratio of data to use for evaluation in SFT mode (default: 0.02)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Loading dataset from {args.input_path}")
    
    if args.mode == "sft":
        # Prepare dataset for SFT training
        dataset = load_from_disk(args.input_path)
        if 'train' in dataset:
            dataset = dataset['train']
        
        processed = prepare_sft_dataset(
            dataset,
            eval_split_ratio=args.eval_split_ratio
        )
        processed.save_to_disk(args.output_path)
        logging.info(f"SFT dataset saved - Train: {len(processed['train'])}, Test: {len(processed['test'])}")
        
    elif args.mode == "rl":
        # Prepare dataset for RL training
        processed = preprocess_grpo_dataset(
            args.input_path,
            split="train",
            enable_thinking=args.enable_thinking
        )
        processed.save_to_disk(args.output_path)
        logging.info(f"RL dataset saved - {len(processed)} examples")
    
    logging.info(f"Processed dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
