# Distillation Demo: What Anthropic Accused DeepSeek of Doing

**For video: "I Replicated What Anthropic Accused DeepSeek of Doing"**

On February 24, 2026, Anthropic published a blog post accusing DeepSeek, MiniMax,
and Moonshot AI of running "industrial-scale distillation campaigns" against Claude,
using 24,000 fraudulent accounts and 16 million exchanges to extract coding and
agentic reasoning capabilities.

This demo recreates the same technique at toy scale (~100 examples instead of 16M)
to show what distillation actually is, how it works, and why the story is more
nuanced than the headlines suggest. We start with 32 hand-written coding prompts
and expand them to ~100 via programmatic variations (adding error handling, tests,
complexity analysis, etc.) — a miniature version of how real campaigns generate
prompt diversity at scale.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Collect training data from Claude (~$1.50-2.00, takes ~5 min)
export ANTHROPIC_API_KEY=your-key-here
python 01_collect_teacher_data.py

# Step 2: Fine-tune a small model on Claude's outputs (~5-15 min on Apple Silicon/GPU)
python 02_finetune_student.py

# Step 3: Compare the models (the money shot)
python 03_compare_models.py
```

## Hardware Requirements

- **Step 1**: Just needs internet access and an Anthropic API key
- **Step 2**: Apple Silicon Macs (M1+) or a GPU with 8+ GB VRAM.
  An M4 Max finishes in ~5-10 minutes. CPU-only is possible but slow (~1-2 hours).
- **Step 3**: Same as Step 2 (runs inference on the local models)

## Video Production Notes

### Pre-recording Checklist

1. **Pre-run Steps 1 and 2** before recording. Step 1 costs ~$1.50-2.00 in API
   calls. Step 2 takes 5-15 minutes on Apple Silicon or a decent GPU.

2. **Terminal font size**: Bump it up to at least 16pt so viewers can read the
   output. Use a dark terminal theme with good contrast.

3. **Have Anthropic's blog post open** in a browser tab for reference:
   https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks

### Suggested Recording Flow

**Intro** (~2 min)
- Show the Anthropic blog headline and the key stat
- "Let me show you exactly what this means — I'm going to do
  the same thing, right now, with one honest account"

**Step 1: Data Collection** (~3 min)
- Walk through `01_collect_teacher_data.py` briefly in your editor
- Point out: the prompts are *targeted* at coding, just like what
  Anthropic described
- Point out the prompt expansion step — "a real campaign generates
  variations programmatically; we do the same thing in miniature"
- Run it live (or show a pre-recorded run). The cost estimate at the end
  is the punchline: "our demo cost two dollars. Scale that up to 16
  million exchanges and..."

**Step 2: Fine-tuning** (~3 min, mostly pre-recorded)
- Show the script briefly. Point out the LoRA config — "we're only
  updating a tiny fraction of the model's parameters"
- Show a pre-recorded training run or the saved output
- "This took 15 minutes on my GPU. The Chinese labs had months."

**Step 3: The Comparison** (~5 min, LIVE)
- This is the part to do live. Run `03_compare_models.py` and let
  the audience see the three outputs side by side.
- The base model (before distillation) will produce mediocre code
- The distilled model will be noticeably better — more structured,
  better variable names, more complete solutions
- It won't match Claude, but the improvement is visible

**The Editorial** (~4 min)
- The irony: Anthropic made money from every one of those exchanges
- The double standard: Anthropic distills its own models (Haiku from
  larger Claude). The AI industry was built on training on others' data.
  The music publishers suing Anthropic would have thoughts about this.
- The real issue: The TOS violations and fraudulent accounts are
  legitimate concerns. But "distillation attack" is a rhetorical
  choice that frames a contract dispute as cyberwarfare.
- The geopolitics: This blog post dropped the same day as a tense
  Pentagon meeting. Anthropic is explicitly lobbying for export controls.

### Title/Thumbnail Ideas

- "I Replicated What Anthropic Accused DeepSeek of Doing"
- "Anthropic Says China 'Stole' Claude — Here's What Actually Happened"
- "I 'Attacked' Claude (With Its Own API)"
- "The $2 Distillation 'Attack' Anthropic Is Warning About"

Thumbnail: Split screen of Claude logo → small model, with
"DISTILLED" stamped across it in red. Or your face looking
skeptical next to the Anthropic blog headline.

## File Overview

| File                         | Purpose                                       |
|------------------------------|-----------------------------------------------|
| `01_collect_teacher_data.py` | Query Claude, save responses as training data |
| `02_finetune_student.py`     | Fine-tune a small model on those responses    |
| `03_compare_models.py`       | Side-by-side comparison (the live demo)       |
| `requirements.txt`           | Python dependencies                           |
| `teacher_data.jsonl`         | Generated by Step 1 (not committed)           |
| `distilled-student/`         | Generated by Step 2 (not committed)           |

## Customization

**Swap the student model** — Use `--model` in Steps 2 and 3. Step 3 reads the
base model from the saved adapter config automatically, so you only need to
specify it once during training:
```bash
python 02_finetune_student.py --model "Qwen/Qwen2.5-1.5B-Instruct"
python 03_compare_models.py   # auto-detects the base model
```

Available models (roughly by size):
- `Qwen/Qwen2.5-0.5B-Instruct` — fastest, fits anywhere
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — slightly better results
- `Qwen/Qwen2.5-1.5B-Instruct` — good balance of quality and speed (default)
- `microsoft/Phi-3-mini-4k-instruct` — best results, needs ~10GB VRAM

**Add more prompts** — The 32 base prompts are expanded to ~100 via
programmatic modifiers (error handling, tests, complexity analysis, etc.).
You can add more base prompts to `CODING_PROMPTS` or add new modifiers to
`PROMPT_MODIFIERS` in `01_collect_teacher_data.py`. At 16 million,
you'd see why Anthropic is concerned.

**Target different capabilities** — We targeted coding, but you could target
creative writing, reasoning, tool use, or anything else. The Chinese labs
allegedly focused on "agentic reasoning, tool use, and coding."

## License

MIT. The irony of licensing a distillation demo is not lost on us.
