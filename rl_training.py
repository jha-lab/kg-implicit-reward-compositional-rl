"""
Reinforcement Learning training using GRPO (Group Relative Policy Optimization).

This script continues training from an SFT checkpoint using RL with multiple
reward signals including knowledge graph path alignment.

Key features:
- Loads and merges SFT LoRA checkpoint
- Multiple reward functions (correctness, path alignment, thinking quality, semantic similarity)
- DeepSpeed ZeRO-3 support for distributed training
- Knowledge graph path integration
"""

import os
import re
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import warnings
import logging
import shutil

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_from_disk, Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

# Import data preprocessing utilities
from data_prep import preprocess_grpo_dataset


# =====================================================================
#                       System Prompt
# =====================================================================
SYSTEM_PROMPT = """A conversation between user and assistant. The user asks a single-choice Multiple Choice Question, and the assistant solves it using step-by-step reasoning. Please answer the multiple choice question by selecting only one from option A, option B, option C, option D. 

The assistant first thinks through the problem systematically, then provides the explanation and final answer. Use <think>...</think> tags for internal reasoning, then provide the explanation process and answer enclosed within <explanation> </explanation> and <answer> </answer> tags, respectively."""

TASK_SPECIFIC_INSTRUCTIONS = "Please provide complete and accurate answers with clear reasoning. The answer must only be a single letter from A, B, C, D."


@dataclass
class TrainingConfig:
    """Configuration for GRPO (RL) training."""
    
    # Model configuration
    model_name: str = field(
        default="Qwen/Qwen3-14B",
        metadata={"help": "Base model name"}
    )
    sft_checkpoint_path: str = field(
        default="./sft_models/model-lora/checkpoint-XXX",
        metadata={"help": "Path to SFT LoRA checkpoint"}
    )
    cache_dir: str = field(
        default="~/.cache/huggingface/hub",
        metadata={"help": "HuggingFace cache directory"}
    )
    
    # Dataset configuration
    dataset_path: str = field(
        default="/path/to/your/rl_dataset",
        metadata={"help": "Path to RL training dataset"}
    )
    
    # Training configuration
    output_dir: str = field(
        default="./rl_models/model-grpo",
        metadata={"help": "Output directory for checkpoints"}
    )
    learning_rate: float = field(
        default=8e-6,
        metadata={"help": "Learning rate (lower than SFT)"}
    )
    beta: float = field(
        default=0.05,
        metadata={"help": "KL penalty coefficient"}
    )
    
    # GRPO-specific parameters
    num_generations: int = field(
        default=2,
        metadata={"help": "Number of completions to generate per prompt"}
    )
    max_prompt_length: int = field(
        default=896,
        metadata={"help": "Maximum prompt length in tokens"}
    )
    max_completion_length: int = field(
        default=896,
        metadata={"help": "Maximum completion length in tokens"}
    )
    
    # Batch configuration
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=32,
        metadata={"help": "Gradient accumulation steps"}
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Number of RL training epochs"}
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "Save checkpoint every N steps"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping"}
    )
    
    # Logging
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "WandB project name"}
    )


# =====================================================================
#                       Utility Functions
# =====================================================================

def extract_answer(text: str) -> str:
    """Extract answer letter (A-D) from model output."""
    try:
        text_clean = re.sub(r'\*+', '', text)
        
        # Look for "Final Answer: X" pattern
        match = re.search(r'Final Answer\s*[:\-]\s*([A-D])', text_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fallback: Check for <answer>X</answer> tags
        answer_match = re.search(r'<answer>\s*([A-D])\s*</answer>', text_clean, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Fallback: Extract from after </think>
        if '</think>' in text_clean:
            after_think = text_clean.split('</think>')[-1]
            letters = re.findall(r'\b[A-D]\b', after_think)
            if letters:
                return letters[0].upper()
        
        return ""
    except Exception:
        return ""


def extract_thinking(text: str) -> str:
    """Extract content within <think> tags."""
    try:
        return text.split("<think>")[-1].split("</think>")[0].strip()
    except IndexError:
        return ""


# =====================================================================
#                       Reward Functions
# =====================================================================

STOP_WORDS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'can', 'this', 'that', 'these', 'those'
}


def normalize_tokens(text: str) -> List[str]:
    """Normalize text to tokens, removing stop words."""
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t and t not in STOP_WORDS]


def repetition_penalty_factor(tokens: List[str], threshold: float = 0.35) -> float:
    """Calculate penalty for repetitive text."""
    if not tokens:
        return 1.0
    
    from collections import Counter
    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    ratio = most_common / max(1, len(tokens))
    
    base = max(0.0, 1.0 - max(0.0, ratio - threshold) * 3.0)
    
    # Penalty for consecutive repeats
    max_run = 1
    current_run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    
    run_penalty = 1.0 - max(0.0, (max_run - 3)) * 0.05
    return max(0.0, base * run_penalty)


def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward function for answer correctness.
    Returns positive reward for correct answers, negative for incorrect/missing.
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_answer(r) for r in responses]
    
    # Extract ground truth answers
    gt_answers = []
    for ans in answer:
        letters = re.findall(r'\b[A-D]\b', ans)
        gt_answers.append(letters[-1] if letters else "")
    
    # Calculate rewards
    rewards = []
    for pred, gt in zip(extracted, gt_answers):
        if pred == gt and pred != "":
            rewards.append(0.1)  # Correct answer
        elif pred == "":
            rewards.append(-1.0)  # No answer
        else:
            rewards.append(-1.0)  # Wrong answer
    
    return rewards


def path_alignment_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward function for knowledge graph path alignment.
    Measures overlap between model reasoning and KG paths.
    """
    responses = [completion[0]["content"] for completion in completions]
    thinkings = [extract_thinking(r) for r in responses]
    
    paths_batch = kwargs.get("paths", [None] * len(responses))
    
    rewards = []
    for thinking, kg_path in zip(thinkings, paths_batch):
        if kg_path is None:
            rewards.append(0.0)
            continue
        
        # Build token sets
        path_tokens = set(normalize_tokens(str(kg_path)))
        thinking_tokens_list = normalize_tokens(thinking)
        thinking_tokens_set = set(thinking_tokens_list)
        
        if not path_tokens:
            rewards.append(0.0)
            continue
        
        # Calculate overlap
        hits = thinking_tokens_set & path_tokens
        coverage = len(hits) / max(1, len(path_tokens))
        min_unique_hit = 1.0 if len(hits) >= 2 else 0.0
        
        # Apply repetition penalty
        rep_factor = repetition_penalty_factor(thinking_tokens_list)
        
        base_reward = (1.2 * coverage + 0.3 * min_unique_hit)
        rewards.append(min(base_reward * rep_factor, 1.5))
    
    return rewards

def thinking_quality_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Simple thinking reward emphasizing stepwise structure; gated on valid answer.
    Range: 0..1.0
    """
    responses = [completion[0]["content"] for completion in completions]
    thinkings = [extract_thinking(r) for r in responses]
    predicted_answers = []
    for r in responses:
        letters = re.findall(r'\b[A-D]\b', extract_answer(r))
        predicted_answers.append(letters[-1] if letters else "")

    step_keywords = ['first', 'second', 'then', 'therefore', 'because', 'thus', 'next', 'finally', 'consider']

    rewards: List[float] = []
    for idx, th in enumerate(thinkings):
        # Soft gate: small negative if no valid extracted answer to keep learning signal
        if not predicted_answers[idx]:
            tokens = normalize_tokens(th)
            rep_factor = repetition_penalty_factor(tokens)
            rewards.append(-0.1 * (1.0 - rep_factor))
            continue
        has_structure = 1.0 if len(th.strip()) >= 20 else 0.0
        step_score = sum(1 for k in step_keywords if k in th.lower()) / max(1, len(step_keywords))
        enum_steps = len(re.findall(r'(^|\n)\s*\d+[)\.-]', th))
        enum_score = min(enum_steps / 3.0, 1.0)
        rep_factor = repetition_penalty_factor(normalize_tokens(th))
        base = (0.5 * has_structure + 0.3 * step_score + 0.2 * enum_score)
        rewards.append(base * rep_factor)
    return rewards



def semantic_answer_similarity_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward semantic overlap between the model's thinking and the ground truth explanation.
    Range: 0..1.0.
    Compares model's <think> reasoning against expert explanation from question_and_explanation field.
    """
    responses = [completion[0]["content"] for completion in completions]
    thinkings = [extract_thinking(r) for r in responses]

    # Get ground truth explanations from dataset
    gt_explanations = kwargs.get("gt_explanation", [None] * len(responses))
    
    rewards: List[float] = []
    for idx, (thinking, gt_explanation) in enumerate(zip(thinkings, gt_explanations)):
        # If no ground truth explanation available, skip
        if not gt_explanation or not gt_explanation.strip():
            rewards.append(0.0)
            continue
        
        # Get model's thinking content
        model_text = thinking.strip()
        if not model_text:
            rewards.append(0.0)
            continue

        # Compute Jaccard similarity between model thinking and ground truth explanation
        gt_tokens = set(normalize_tokens(gt_explanation))
        model_tokens_list = normalize_tokens(model_text)
        model_tokens = set(model_tokens_list)
        
        if not gt_tokens or not model_tokens:
            rewards.append(0.0)
            continue

        intersection = len(gt_tokens & model_tokens)
        union = len(gt_tokens | model_tokens)
        jaccard = intersection / max(1, union)
        rep_factor = repetition_penalty_factor(model_tokens_list)
        rewards.append(jaccard * rep_factor)
        print(f"Explanation: {gt_explanation[:200]}")
        print(f"Jaccard similarity: {jaccard}")
        print(f"Repetition penalty factor: {rep_factor}")
        print(f"Reward: {jaccard * rep_factor}")
        print("--------------------------------")
    



# =====================================================================
#                       Main Training Function
# =====================================================================

def train():
    """Main RL training function."""
    
    # Setup environment variables
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
    
    # Parse configuration
    parser = transformers.HfArgumentParser(TrainingConfig)
    config = parser.parse_args_into_dataclasses()[0]
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize distributed training
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            import datetime
            dist.init_process_group(
                backend='nccl',
                timeout=datetime.timedelta(seconds=3600)
            )
        dist.barrier()
    
    logging.info(f"Rank {local_rank}/{world_size} initialized")
    
    # Load and preprocess dataset for GRPO training
    logging.info(f"Preprocessing dataset from {config.dataset_path}")
    dataset = preprocess_grpo_dataset(
        dataset_path=config.dataset_path,
        split="train",
        chunk_size=1000,
        enable_thinking=True,
        system_prompt=SYSTEM_PROMPT,
        task_instructions=TASK_SPECIFIC_INSTRUCTIONS
    )
    logging.info(f"Dataset preprocessed - {len(dataset)} examples")
    
    # Load base model and merge SFT adapters (rank 0 only)
    is_rank0 = (world_size == 1) or (dist.get_rank() == 0)
    
    if is_rank0:
        logging.info("Loading base model and merging SFT adapters...")
        
        # Get base model path from adapter config
        import json
        adapter_config_path = os.path.join(config.sft_checkpoint_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config["base_model_name_or_path"]
        
        # Load base model on CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Load and merge LoRA adapters
        peft_model = PeftModel.from_pretrained(base_model, config.sft_checkpoint_path)
        merged_model = peft_model.merge_and_unload()
        del peft_model, base_model
        
        # Save merged model
        merged_path = os.path.join(config.output_dir, "temp_merged_model")
        os.makedirs(merged_path, exist_ok=True)
        merged_model.save_pretrained(merged_path, safe_serialization=True)
        
        # Write readiness marker
        with open(os.path.join(merged_path, "_READY"), "w") as f:
            f.write("ready")
        
        del merged_model
        torch.cuda.empty_cache()
        logging.info("Model merged and saved")
    
    # Synchronize
    if world_size > 1:
        dist.barrier()
    
    # All ranks load merged model
    merged_path = os.path.join(config.output_dir, "temp_merged_model")
    
    # Wait for readiness marker (non-rank0)
    if world_size > 1 and not is_rank0:
        import time
        while not os.path.exists(os.path.join(merged_path, "_READY")):
            time.sleep(1.0)
    
    logging.info("Loading merged model for RL training...")
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        use_cache=False,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_checkpoint_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure GRPO training
    training_args = GRPOConfig(
        learning_rate=config.learning_rate,
        beta=config.beta,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,
        logging_steps=1,
        bf16=True,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_prompt_length + config.max_completion_length,
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.9,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.15,
        },
        optim="adamw_torch",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        save_steps=config.save_steps,
        max_grad_norm=config.max_grad_norm,
        output_dir=config.output_dir,
        report_to=[] if config.wandb_project is None else ["wandb"],
    )
    
    # Define reward functions
    # All four reward functions are available - uncomment as needed for your use case
    reward_funcs = [
        correctness_reward_func,
        path_alignment_reward_func,
    ]
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    logging.info("Starting GRPO training")
    trainer.train()
    trainer.save_model(config.output_dir)
    trainer.accelerator.wait_for_everyone()
    
    # Cleanup temp directory
    if is_rank0:
        if os.path.exists(merged_path):
            shutil.rmtree(merged_path)
            logging.info("Cleaned up temporary merged model")
    
    logging.info("Training complete!")


if __name__ == "__main__":
    train()
