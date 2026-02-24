"""
Step 3: Compare Teacher vs Base Student vs Distilled Student
============================================================

This is the money shot for the video. We compare three models on
the same prompt:

  1. The Teacher (Claude) — the frontier model being distilled
  2. The Base Student — the small model BEFORE distillation
  3. The Distilled Student — the same small model AFTER training
     on Claude's outputs

The visual of a 0.5B model producing Claude-like coding responses
is the "aha moment" that makes distillation click for the audience.

Usage:
    python 03_compare_models.py
    python 03_compare_models.py --prompt "Write a Python binary search"
    python 03_compare_models.py --no-teacher   # skip API call
    python 03_compare_models.py --model "Qwen/Qwen2.5-1.5B-Instruct"  # override base model
"""

import argparse
import json
import os
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Configuration ────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DISTILLED_MODEL_DIR = "./distilled-student"


def resolve_base_model(cli_override=None):
    """Determine which base model to use.

    Priority: CLI --model flag > adapter_config.json > DEFAULT_MODEL fallback.
    """
    if cli_override:
        return cli_override

    adapter_config_path = os.path.join(DISTILLED_MODEL_DIR, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            config = json.load(f)
        base = config.get("base_model_name_or_path")
        if base:
            print(f"  (base model read from {adapter_config_path})")
            return base

    return DEFAULT_MODEL

# Test prompts — should be SIMILAR to training prompts but NOT identical.
# This tests whether the student actually learned the capability vs memorizing.
DEFAULT_PROMPTS = [
    "Write a Python function to check if a binary tree is balanced. Include type hints.",
    "Implement a simple connection pool in Python that limits the number of concurrent connections.",
    "Write a SQL query using window functions to find employees whose salary is above their department average.",
    "Explain how virtual threads in Java differ from platform threads, with a practical example.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare teacher vs student models")
    parser.add_argument("--prompt", type=str, default=None, help="Custom test prompt")
    parser.add_argument("--no-teacher", action="store_true", help="Skip teacher API call")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--model", type=str, default=None,
                        help="Base student model (default: read from adapter config)")
    return parser.parse_args()


def get_teacher_response(prompt, max_tokens=512):
    """Get the teacher's response via API."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _detect_dtype_and_device():
    """Pick dtype and device consistently across load functions."""
    if torch.cuda.is_available():
        return torch.bfloat16, "auto"
    elif torch.backends.mps.is_available():
        return torch.float16, None  # MPS needs explicit .to("mps")
    else:
        return torch.float32, None


def _move_to_mps_if_needed(model):
    """Move model to MPS when CUDA isn't available but MPS is."""
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        model = model.to("mps")
    return model


def load_base_model(base_model_name):
    """Load the student model WITHOUT distillation."""
    print(f"Loading base model: {base_model_name}")
    dtype, device_map = _detect_dtype_and_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    model = _move_to_mps_if_needed(model)
    return model, tokenizer


def load_distilled_model(base_model_name):
    """Load the student model WITH distillation (LoRA adapter)."""
    print(f"Loading distilled model from: {DISTILLED_MODEL_DIR}")
    dtype, device_map = _detect_dtype_and_device()
    tokenizer = AutoTokenizer.from_pretrained(DISTILLED_MODEL_DIR, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, DISTILLED_MODEL_DIR)
    model = model.merge_and_unload()  # Merge LoRA weights for faster inference
    model = _move_to_mps_if_needed(model)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a response from a local model."""
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

    # Decode only the generated tokens (not the prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def print_section(title, content, width=78):
    """Pretty-print a section for the video."""
    border = "─" * width
    print(f"\n┌{border}┐")
    print(f"│ {title:<{width-1}}│")
    print(f"├{border}┤")
    for line in content.split("\n"):
        # Wrap long lines
        wrapped = textwrap.wrap(line, width=width - 2) or [""]
        for wline in wrapped:
            print(f"│ {wline:<{width-1}}│")
    print(f"└{border}┘")


def run_comparison(prompt, args, base_model_name):
    """Run a side-by-side comparison for one prompt."""
    print("\n" + "=" * 80)
    print(f"TEST PROMPT: {prompt}")
    print("=" * 80)

    # ── Teacher (Claude) ─────────────────────────────────────────
    if not args.no_teacher:
        print("\n⏳ Querying teacher (Claude)...")
        teacher_response = get_teacher_response(prompt, args.max_tokens)
        print_section("🎓 TEACHER (Claude Sonnet — hundreds of billions of params)",
                      teacher_response[:1500])
    else:
        print("\n⏭️  Skipping teacher (--no-teacher flag)")

    # ── Base Student (before distillation) ───────────────────────
    print("\n⏳ Generating from base student (BEFORE distillation)...")
    base_model, base_tokenizer = load_base_model(base_model_name)
    base_response = generate_response(base_model, base_tokenizer, prompt, args.max_tokens)
    short_name = base_model_name.split("/")[-1]
    print_section(f"📝 BASE STUDENT ({short_name} — no distillation)",
                  base_response[:1500])

    # Free memory before loading next model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Distilled Student (after distillation) ───────────────────
    print("\n⏳ Generating from distilled student (AFTER distillation)...")
    try:
        dist_model, dist_tokenizer = load_distilled_model(base_model_name)
        dist_response = generate_response(dist_model, dist_tokenizer, prompt, args.max_tokens)
        print_section(f"🎯 DISTILLED STUDENT ({short_name} — trained on Claude's outputs)",
                      dist_response[:1500])
        del dist_model
    except Exception as e:
        print(f"\n⚠️  Could not load distilled model: {e}")
        print("   Have you run 02_finetune_student.py first?")
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    base_model_name = resolve_base_model(args.model)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           DISTILLATION DEMO — MODEL COMPARISON              ║
║                                                              ║
║  Showing how a small model can learn to mimic Claude's      ║
║  coding ability by training on its outputs.                  ║
║                                                              ║
║  Student model: {base_model_name:<43}║
║  This is what Anthropic accused DeepSeek, MiniMax, and      ║
║  Moonshot of doing at industrial scale (16M exchanges).     ║
║  We're doing the same thing with ~100 examples.              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    for prompt in prompts:
        run_comparison(prompt, args, base_model_name)

    print("\n" + "=" * 80)
    print("TAKEAWAYS FOR THE VIDEO:")
    print("=" * 80)
    print("""
1. The distilled student isn't as good as Claude — but it's dramatically
   better than the base model on the targeted capability.

2. We did this with ~100 training examples. The Chinese labs used 16 MILLION.
   At that scale, the capability transfer is substantial.

3. Every one of those 16 million exchanges was a paid API call.
   Anthropic made money from the "attack."

4. This is the SAME technique Anthropic uses to create Claude Haiku
   from larger Claude models. Distillation is standard practice.
   The question is: who gets to do it, and to whom?
""")


if __name__ == "__main__":
    main()
