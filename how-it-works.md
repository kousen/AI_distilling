# How It Works — Technical Deep Dive

A reference guide for how each step of the distillation pipeline works under the hood.

---

## Step 1: Collecting Teacher Data via the Anthropic API

### The API Surface

The entire data collection step uses a single Anthropic API call, repeated 104 times:

```python
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)

completion = response.content[0].text
```

That's it. There's no special "distillation endpoint" — it's the same Messages API that any developer uses to build a chatbot or coding assistant. Each call sends one user message and gets back one assistant response.

### What Makes It "Distillation"

The magic isn't in the API call — it's in what you *do* with the responses. Instead of showing them to a user, you save them as training data:

```python
entry = {
    "prompt": prompt,           # what we asked
    "completion": completion,   # what Claude said
}
# Written to teacher_data.jsonl, one JSON object per line
```

This creates prompt-completion pairs that capture Claude's *behavior* — its coding style, error handling patterns, documentation conventions, and reasoning approach. The student model will later learn to imitate these patterns.

### Prompt Expansion

32 hand-written base prompts aren't enough to reliably shift a model's behavior. The script applies 6 modifiers to subsets of the base prompts:

| Modifier | Applied To | Purpose |
|---|---|---|
| "Add comprehensive error handling..." | Prompts 0-15 | Elicit Claude's error handling patterns |
| "Include unit tests using pytest..." | Prompts 0-15 | Capture testing style |
| "Analyze the time and space complexity..." | Prompts 0-7 | Extract reasoning about algorithms |
| "Implement an alternative approach..." | Prompts 0-7, 20-23 | Get multiple solutions per problem |
| "Walk through your reasoning step by step..." | Prompts 8-19 | Capture chain-of-thought patterns |
| "Make this production-ready..." | Prompts 8-15 | Elicit production coding conventions |

This takes 32 base prompts → 104 unique prompts. Each modifier elicits a genuinely different response from Claude, so the training data covers more of the target capability surface.

A real distillation campaign would generate thousands or millions of variations programmatically. The Chinese labs allegedly used 16 million exchanges.

### Cost

At Sonnet pricing ($3/M input tokens, $15/M output tokens), our 104 exchanges cost ~$1.61. Every one of those tokens is revenue for Anthropic — the "attack" paid the victim.

---

## Step 2: Fine-Tuning with LoRA

### The Big Picture

Fine-tuning means taking a pre-trained model and continuing its training on new data. The model has already learned language from billions of tokens of internet text. We're nudging its behavior on a narrow task (coding) by showing it how Claude responds to coding prompts.

### Why LoRA Instead of Full Fine-Tuning

A 1.5B parameter model has 1.5 billion floating-point numbers that define its behavior. Full fine-tuning would update all of them, which:
- Requires storing a full copy of all gradients (doubles memory)
- Is slow (every parameter gets an update each step)
- Risks catastrophic forgetting (the model forgets what it already knew)

**LoRA (Low-Rank Adaptation)** is a shortcut. Instead of updating the full weight matrices, it adds small "adapter" matrices alongside the existing weights:

```
Original: output = input × W          (W is huge, e.g., 1536×1536)
LoRA:     output = input × W + input × A × B   (A is 1536×16, B is 16×1536)
```

The original weight matrix `W` is frozen. Only `A` and `B` are trained. With rank `r=16`, this means:
- Original matrix: 1,536 × 1,536 = **2.36M parameters**
- LoRA matrices: (1,536 × 16) + (16 × 1,536) = **49K parameters** (2% of original)

Applied across the attention layers (q_proj, v_proj, k_proj, o_proj), the total trainable parameters are 1.7M out of 1.5B — just **0.1%** of the model.

### The LoRA Configuration

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # we're training a text generator
    r=16,                             # rank of the low-rank matrices
    lora_alpha=32,                    # scaling factor (alpha/r = effective learning rate multiplier)
    lora_dropout=0.05,                # dropout for regularization
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # which layers to adapt
)
```

**Why these target modules?** The attention mechanism is where the model decides "what to pay attention to" when generating each token. By adapting the query, key, value, and output projections, we're changing *how the model attends* to the input — which is enough to shift its coding style without touching the feedforward layers that store factual knowledge.

**Why r=16?** The rank controls the capacity of the adaptation. Higher rank = more parameters = more expressive but slower and more memory. For our 104-example dataset, r=16 is plenty — we're transferring *style*, not teaching new facts.

**lora_alpha=32 with r=16**: The effective scaling is alpha/r = 2.0, which means the LoRA updates are amplified by 2x relative to the base weights. This helps the small adapter matrices have a meaningful impact.

### Data Preparation

Before training, each prompt-completion pair must be formatted into the student model's expected chat template:

```python
messages = [
    {"role": "user", "content": item["prompt"]},
    {"role": "assistant", "content": item["completion"]},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

For Qwen 2.5, this produces something like:

```
<|im_start|>user
Write a Python function to find the longest common subsequence...
<|im_end|>
<|im_start|>assistant
Here's an implementation of the LCS algorithm...
<|im_end|>
```

The special tokens (`<|im_start|>`, `<|im_end|>`) tell the model where each turn begins and ends. Every model family has its own template — using the wrong one means the model can't parse the training data properly.

### Training Configuration

```python
training_args = SFTConfig(
    num_train_epochs=5,                    # pass through the data 5 times
    per_device_train_batch_size=2,         # 2 examples per step
    gradient_accumulation_steps=4,         # accumulate 4 steps before updating
    learning_rate=2e-4,                    # how big each update is
    weight_decay=0.01,                     # regularization to prevent overfitting
    warmup_ratio=0.1,                      # ramp up LR for the first 10% of steps
    dataset_text_field="text",             # which field in the dataset has the text
    max_length=1024,                       # truncate sequences longer than this
)
```

Key relationships:
- **Effective batch size** = batch_size × gradient_accumulation = 2 × 4 = **8 examples per update**
- **Steps per epoch** = ceil(104 / 8) = **13 steps**
- **Total training steps** = 13 × 5 = **65 steps** (tiny by ML standards)
- **Warmup steps** = 65 × 0.1 ≈ **6 steps** (learning rate ramps from 0 to 2e-4)

### What Happens During Training

Each training step:

1. **Forward pass**: Feed a batch of formatted text through the model. The model predicts the next token at each position.
2. **Loss calculation**: Compare the model's predictions to the actual next tokens (from Claude's responses). The loss measures how wrong the predictions are.
3. **Backward pass**: Compute gradients — how much each LoRA parameter should change to reduce the loss.
4. **Parameter update**: Adjust only the LoRA matrices A and B (the base model weights W stay frozen).

The loss dropped from 0.900 to 0.809 over 5 epochs. Token accuracy (how often the student's top prediction matches Claude's next token) went from 78.1% to 79.6%, peaking at 82.2%.

### What the Student Actually Learns

The student doesn't learn to *be* Claude. It learns to produce text that *looks like* Claude's text in the specific domain we trained on. Concretely:
- **Structure patterns**: dataclasses, custom exception hierarchies, context managers
- **Documentation style**: section headers, docstrings with Examples sections, type hints
- **Error handling vocabulary**: comprehensive try/except, custom exceptions, validation
- **Design patterns**: factory methods, builder patterns, proper OOP structure

It does NOT effectively learn:
- Claude's factual knowledge (the 1.5B model's knowledge is fixed)
- Claude's reasoning depth (the smaller model can't sustain long chains of thought)
- Claude's ability to handle novel problems (it learned patterns, not problem-solving)

### The Output

Training produces a `distilled-student/` directory containing:
- `adapter_model.safetensors` (17 MB) — the LoRA weights
- `adapter_config.json` — points back to the base model and LoRA settings
- Tokenizer files (copied from the base model)

The adapter is tiny because it only contains the LoRA matrices. At inference time, the adapter is loaded on top of the base model:

```python
# Load base model (1.5B parameters, ~3 GB)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Load adapter (1.7M parameters, ~17 MB) on top
model = PeftModel.from_pretrained(model, "./distilled-student")
```

The base model weights are unchanged. The LoRA matrices are added to the attention layers, slightly shifting how the model generates text.

---

## Step 3: Comparing the Models

Step 3 loads three models and runs them on the same test prompts:

1. **Teacher (Claude)**: Fresh API call to Claude Sonnet 4.6
2. **Base student**: Qwen 2.5 1.5B with no fine-tuning
3. **Distilled student**: Same Qwen model with the LoRA adapter loaded

The comparison makes the distillation effect visible: the base model writes simple, sometimes buggy code, while the distilled model writes "Claude-shaped" code — over-engineered, heavily documented, with custom exception hierarchies and design patterns it picked up from the training data.

---

## The Scale Argument

Everything above took 104 API calls ($1.61) and 11 minutes of local training. The effect is visible but modest.

The Chinese labs allegedly used 16 million exchanges over months. At that scale:
- The training data covers vastly more of Claude's capability surface
- More epochs and larger models absorb more of the teacher's patterns
- The resulting model would be substantially more capable

The technique is identical. Only the scale differs.
