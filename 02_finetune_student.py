"""
Step 2: Fine-Tune a Small Student Model on Teacher Data
=======================================================

Now we take the responses we collected from Claude and use them to
train a much smaller model. This is the "distillation" step.

The student model (1-3B parameters) learns to mimic the teacher's
(Claude, hundreds of billions of parameters) behavior on the specific
capability we targeted — in our case, coding.

We use LoRA (Low-Rank Adaptation) so this can run on a single GPU
or even a MacBook with enough patience.

Usage:
    python 02_finetune_student.py [--epochs 5] [--batch-size 2]

Prerequisites:
    - teacher_data.jsonl from Step 1
    - A GPU with 8+ GB VRAM (or CPU, but much slower)
"""

import argparse
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── Configuration ────────────────────────────────────────────────

# Pick your student model. Smaller = faster to demo, larger = better results.
# Options (roughly by size):
#   "Qwen/Qwen2.5-0.5B-Instruct"   — 0.5B, very fast, fits anywhere
#   "TinyLlama/TinyLlama-1.1B-Chat-v1.0" — 1.1B, good balance
#   "Qwen/Qwen2.5-1.5B-Instruct"   — 1.5B, better quality
#   "microsoft/Phi-3-mini-4k-instruct"    — 3.8B, best quality, needs more VRAM
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TEACHER_DATA = "teacher_data.jsonl"
OUTPUT_DIR = "./distilled-student"

# ── Argument Parsing ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune student on teacher data")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--model", type=str, default=STUDENT_MODEL, help="Student model name")
    return parser.parse_args()


# ── Data Preparation ─────────────────────────────────────────────

def load_and_format_data(tokenizer, data_file=TEACHER_DATA):
    """Load the teacher data and format it for the student model's chat template."""

    with open(data_file) as f:
        raw_data = [json.loads(line) for line in f]

    print(f"Loaded {len(raw_data)} examples from {data_file}")

    formatted = []
    for item in raw_data:
        # Use the student model's chat template if available
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["completion"]},
        ]

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # Fallback: simple instruction format
            text = (
                f"### Instruction:\n{item['prompt']}\n\n"
                f"### Response:\n{item['completion']}"
            )

        formatted.append({"text": text})

    return Dataset.from_list(formatted)


# ── Model Setup ──────────────────────────────────────────────────

def setup_model_and_tokenizer(model_name):
    """Load the student model with LoRA for efficient fine-tuning."""

    print(f"Loading student model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect device
    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device_info = "Apple MPS"
        dtype = torch.float16
    else:
        device_info = "CPU (this will be slow!)"
        dtype = torch.float32

    print(f"Device: {device_info}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # device_map="auto" only works for CUDA; move to MPS explicitly
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        model = model.to("mps")

    # LoRA: instead of updating all 500M+ parameters, we add small
    # trainable adapters. This makes fine-tuning reasonable on consumer hardware.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,               # Rank of the low-rank matrices
        lora_alpha=32,       # Scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Parameters: {total:,} total, {trainable:,} trainable "
          f"({100 * trainable / total:.1f}%)")

    return model, tokenizer


# ── Training ─────────────────────────────────────────────────────

def train(args):
    model, tokenizer = setup_model_and_tokenizer(args.model)
    dataset = load_and_format_data(tokenizer)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=(torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False),
        report_to="none",  # Set to "tensorboard" if you want to show training curves
        optim="adamw_torch",
        dataset_text_field="text",
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("TRAINING START")
    print(f"  Student model:  {args.model}")
    print(f"  Training examples: {len(dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60 + "\n")

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — model saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    train(args)
