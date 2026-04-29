"""
LoRA-Lens: Training Script
===========================
Train LoRA adapters with configurable rank, target modules, and data size.
Designed to run on Colab T4 or Apple Silicon CPU.
"""

import os
import json
import time
import torch
from dataclasses import dataclass, asdict
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class LoRATrainConfig:
    """Training configuration for a single LoRA experiment."""
    config_name: str = "R16"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "tatsu-lab/alpaca"
    max_train_samples: int = 1000

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None  # Default: ["q_proj", "v_proj"]

    # Training hyperparameters
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    seed: int = 42

    # Output
    output_dir: str = "./results/adapters"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


def auto_device() -> str:
    """Select best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # Skip MPS — unstable with many architectures


def auto_dtype(device: str):
    """Select dtype based on device."""
    if device == "cuda":
        # Check if bf16 is supported
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def format_alpaca(example: dict, tokenizer, max_length: int = 512) -> dict:
    """
    Format Alpaca-style dataset into chat template.

    Alpaca format: {instruction, input, output}
    → Chat: [{"role": "user", "content": instruction + input},
             {"role": "assistant", "content": output}]
    """
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp:
        user_content = f"{instruction}\n\nInput: {inp}"
    else:
        user_content = instruction

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    # Use chat template if available
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        text = f"### Instruction:\n{user_content}\n\n### Response:\n{output}"

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def train_lora(config: LoRATrainConfig) -> str:
    """
    Train a LoRA adapter and save it.

    Returns:
        Path to the saved adapter directory.
    """
    device = auto_device()
    dtype = auto_dtype(device)
    print(f"\n{'='*60}")
    print(f"Training: {config.config_name}")
    print(f"Model: {config.model_name}")
    print(f"LoRA rank={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Targets: {config.target_modules}")
    print(f"Data: {config.max_train_samples} samples")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format dataset
    dataset = load_dataset(config.dataset_name, split="train")
    if config.max_train_samples < len(dataset):
        dataset = dataset.shuffle(seed=config.seed).select(range(config.max_train_samples))

    dataset = dataset.map(
        lambda x: format_alpaca(x, tokenizer, config.max_seq_length),
        remove_columns=dataset.column_names,
    )

    # Training arguments
    save_dir = os.path.join(config.output_dir, config.config_name)
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        seed=config.seed,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Save adapter + config
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save training metadata
    meta = {
        "config": asdict(config),
        "train_time_s": round(train_time, 1),
        "device": device,
        "dtype": str(dtype),
        "num_trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved adapter to {save_dir} ({train_time:.0f}s)")
    return save_dir


def load_trained_adapter(
    adapter_dir: str,
    model_name: Optional[str] = None,
):
    """
    Load a trained LoRA adapter for analysis or inference.

    Returns:
        (model, tokenizer)
    """
    from peft import PeftModel

    # Load metadata to get model name
    meta_path = os.path.join(adapter_dir, "train_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if model_name is None:
            model_name = meta["config"]["model_name"]

    device = auto_device()
    dtype = auto_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    print(f"✓ Loaded adapter from {adapter_dir}")
    return model, tokenizer


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_rank_ablation(
    ranks: list[int] = [4, 8, 16, 32, 64],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_train_samples: int = 1000,
    target_modules: list[str] = None,
    output_dir: str = "./results/adapters",
) -> list[str]:
    """
    Run Experiment 1: Train adapters with different ranks.
    Returns list of adapter directory paths.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    adapter_dirs = []
    for r in ranks:
        config = LoRATrainConfig(
            config_name=f"R{r}",
            model_name=model_name,
            lora_r=r,
            lora_alpha=r * 2,  # Standard heuristic: alpha = 2r
            max_train_samples=max_train_samples,
            target_modules=target_modules,
            output_dir=output_dir,
        )
        path = train_lora(config)
        adapter_dirs.append(path)

        # Clear GPU memory between runs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()

    return adapter_dirs


def run_data_scaling(
    sample_sizes: list[int] = [100, 500, 1000, 2000, 5000],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    lora_r: int = 16,
    output_dir: str = "./results/adapters",
) -> list[str]:
    """
    Run Experiment 3: Train adapters with different data sizes.
    """
    adapter_dirs = []
    for n in sample_sizes:
        config = LoRATrainConfig(
            config_name=f"D{n}",
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_r * 2,
            max_train_samples=n,
            output_dir=output_dir,
        )
        path = train_lora(config)
        adapter_dirs.append(path)

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()

    return adapter_dirs


def run_target_module_ablation(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    lora_r: int = 16,
    max_train_samples: int = 1000,
    output_dir: str = "./results/adapters",
) -> list[str]:
    """
    Run Experiment 4: Train adapters targeting different module sets.
    """
    configs = {
        "QV": ["q_proj", "v_proj"],
        "QKVO": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ALL": ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
    }

    adapter_dirs = []
    for name, modules in configs.items():
        config = LoRATrainConfig(
            config_name=f"TM_{name}",
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_r * 2,
            max_train_samples=max_train_samples,
            target_modules=modules,
            output_dir=output_dir,
        )
        path = train_lora(config)
        adapter_dirs.append(path)

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()

    return adapter_dirs
