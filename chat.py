"""
Interactive Chat: Compare Base vs Distilled Student
====================================================

Loads both models once, then lets you submit prompts interactively
and see how the base and distilled models respond side by side.

Usage:
    python chat.py
    python chat.py --model "Qwen/Qwen2.5-1.5B-Instruct"
    python chat.py --max-tokens 256
"""

import argparse
import json
import os
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DISTILLED_MODEL_DIR = "./distilled-student"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_base_model(cli_override=None):
    if cli_override:
        return cli_override
    adapter_config_path = os.path.join(DISTILLED_MODEL_DIR, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            config = json.load(f)
        base = config.get("base_model_name_or_path")
        if base:
            return base
    return DEFAULT_MODEL


def detect_dtype_and_device():
    if torch.cuda.is_available():
        return torch.bfloat16, "auto"
    elif torch.backends.mps.is_available():
        return torch.float16, None
    else:
        return torch.float32, None


def move_to_mps_if_needed(model):
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        model = model.to("mps")
    return model


def load_models(base_model_name):
    dtype, device_map = detect_dtype_and_device()

    print(f"Loading base model: {base_model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=dtype, trust_remote_code=True, device_map=device_map,
    )
    base_model = move_to_mps_if_needed(base_model)

    print(f"Loading distilled model from: {DISTILLED_MODEL_DIR}")
    dist_tokenizer = AutoTokenizer.from_pretrained(DISTILLED_MODEL_DIR, trust_remote_code=True)
    dist_base = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=dtype, trust_remote_code=True, device_map=device_map,
    )
    dist_model = PeftModel.from_pretrained(dist_base, DISTILLED_MODEL_DIR)
    dist_model = dist_model.merge_and_unload()
    dist_model = move_to_mps_if_needed(dist_model)

    return base_model, base_tokenizer, dist_model, dist_tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    elif torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    return response.strip()


def print_section(title, content, width=78):
    border = "─" * width
    print(f"\n┌{border}┐")
    print(f"│ {title:<{width-1}}│")
    print(f"├{border}┤")
    for line in content.split("\n"):
        wrapped = textwrap.wrap(line, width=width - 2) or [""]
        for wline in wrapped:
            print(f"│ {wline:<{width-1}}│")
    print(f"└{border}┘")


def display_responses(base_response, dist_response, short_name="student"):
    print_section(f"📝 BASE ({short_name} — no distillation)", base_response)
    print_section(f"🎯 DISTILLED ({short_name} — trained on Claude's outputs)", dist_response)


def main():
    parser = argparse.ArgumentParser(description="Interactive base vs distilled chat")
    parser.add_argument("--model", type=str, default=None, help="Base student model")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    args = parser.parse_args()

    base_model_name = resolve_base_model(args.model)

    print("Loading models (this takes a moment)...\n")
    base_model, base_tok, dist_model, dist_tok = load_models(base_model_name)

    short_name = base_model_name.split("/")[-1]
    print(f"\nModels loaded. Type a prompt and press Enter.")
    print(f"  Base:      {short_name}")
    print(f"  Distilled: {short_name} + LoRA adapter")
    print(f"Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            print("Bye!")
            break

        print(f"\nGenerating from base model...")
        base_response = generate(base_model, base_tok, prompt, args.max_tokens)

        print(f"Generating from distilled model...")
        dist_response = generate(dist_model, dist_tok, prompt, args.max_tokens)

        display_responses(base_response, dist_response, short_name)


if __name__ == "__main__":
    main()
