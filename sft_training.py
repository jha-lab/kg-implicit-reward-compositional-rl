"""
Supervised Fine-Tuning (SFT) with LoRA for large language models.

This script trains a base model using parameter-efficient fine-tuning (LoRA)
on knowledge graph-guided question answering data.

Supports:
- LoRA (Low-Rank Adaptation) for memory-efficient training
- DeepSpeed ZeRO-3 for distributed training
- Chat template formatting for instruction tuning
- Configurable for any HuggingFace model
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from datasets import load_from_disk, DatasetDict
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer
import trl
import torch

# Import data preprocessing utilities
from data_prep import to_messages_format

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class TrainingConfig:
    """Configuration for SFT training."""
    
    # Model configuration
    model_name: str = field(
        default="Qwen/Qwen3-14B",
        metadata={"help": "HuggingFace model name or path"}
    )
    cache_dir: str = field(
        default="~/.cache/huggingface/hub",
        metadata={"help": "Cache directory for HuggingFace models"}
    )
    
    # Dataset configuration
    dataset_path: str = field(
        default="/path/to/your/tokenized_dataset",
        metadata={"help": "Path to preprocessed dataset"}
    )
    
    # Training configuration
    block_size: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    output_dir: str = field(
        default="./sft_models/model-lora",
        metadata={"help": "Output directory for model checkpoints"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=32,
        metadata={"help": "Gradient accumulation steps"}
    )
    num_train_epochs: int = field(
        default=20,
        metadata={"help": "Number of training epochs"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every N steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file"}
    )
    
    # LoRA configuration
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for parameter-efficient training"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        metadata={"help": "Target modules for LoRA"}
    )
    
    # Logging configuration
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "WandB project name (optional)"}
    )


def load_dataset(config: TrainingConfig) -> DatasetDict:
    """
    Load and prepare the training dataset.
    
    Args:
        config: Training configuration
    
    Returns:
        DatasetDict with 'train' and 'test' splits in messages format
    """
    dataset = load_from_disk(config.dataset_path)
    
    # Ensure dataset has required splits
    if 'train' not in dataset:
        raise ValueError("Dataset must contain a 'train' split")
    
    if 'test' not in dataset:
        logging.warning("No test split found, using 2% of train for evaluation")
        train_size = len(dataset['train'])
        test_size = max(1, int(0.02 * train_size))
        test_dataset = dataset['train'].select(range(test_size))
        train_dataset = dataset['train'].select(range(test_size, train_size))
        dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    
    # Convert to messages format for TRL training
    logging.info("Converting dataset to messages format...")
    dataset = DatasetDict({
        'train': dataset['train'].map(to_messages_format, batched=False),
        'test': dataset['test'].map(to_messages_format, batched=False),
    })
    
    logging.info(f"Loaded dataset - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    return dataset


def train():
    """Main training function."""
    
    # Parse configuration
    parser = HfArgumentParser(TrainingConfig)
    config = parser.parse_args_into_dataclasses()[0]
    
    # Disable WandB if not configured
    if config.wandb_project is None:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Load dataset
    dataset = load_dataset(config)
    
    # Load tokenizer
    logging.info(f"Loading tokenizer from {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    
    # Load model
    logging.info(f"Loading model {config.model_name}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "use_cache": False,
        "cache_dir": config.cache_dir,
    }
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    # Setup chat template if needed
    if getattr(tokenizer, "chat_template", None) is None:
        try:
            # Try to get chat template from a reference model
            from trl import clone_chat_template
            model, tokenizer, _ = clone_chat_template(model, tokenizer, config.model_name)
            logging.info("Successfully set chat template")
        except Exception as e:
            logging.warning(f"Could not set chat template: {e}")
    
    # Ensure EOS token is set
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.eos_token = "</s>"
    
    # Resize token embeddings if we added special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    lora_config: Optional[LoraConfig] = None
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        logging.info(f"LoRA config: rank={config.lora_r}, alpha={config.lora_alpha}")
    
    # Configure SFT training arguments
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        deepspeed=config.deepspeed,
        save_total_limit=3,
        logging_first_step=True,
        report_to=[] if config.wandb_project is None else ["wandb"],
        max_seq_length=config.block_size,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    logging.info("Initializing SFT trainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        args=sft_config,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    # Train
    logging.info("Starting training")
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    
    # Save final model
    os.makedirs(config.output_dir, exist_ok=True)
    trainer.save_model(output_dir=config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logging.info(f"Training complete! Model saved to {config.output_dir}")


if __name__ == "__main__":
    train()
