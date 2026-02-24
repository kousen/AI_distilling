# Distillation Demo — Test Run Results

Run date: 2026-02-24
Hardware: Apple M4 Max, MPS (Metal Performance Shaders)

## Step 1: Data Collection (`01_collect_teacher_data.py`)

- **Prompts**: 104/104 successful (32 base + 72 expanded variations)
- **Time**: ~24 minutes 40 seconds
- **Cost**: ~$1.61
- **Teacher model**: `claude-sonnet-4-6`
- **Output**: `teacher_data.jsonl`

## Step 2: Fine-tuning (`02_finetune_student.py`)

- **Student model**: `Qwen/Qwen2.5-0.5B-Instruct` (494M params, 1.7M trainable via LoRA = 0.3%)
- **Time**: ~2 minutes 55 seconds
- **Loss**: 1.289 → 0.991
- **Accuracy**: 72.7% → 76.0%
- **Epochs**: 3
- **Output**: `distilled-student/` directory

## Step 3: Comparison (`03_compare_models.py --no-teacher`)

Four test prompts, each comparing the base student (before distillation) against
the distilled student (after training on Claude's outputs). Teacher (Claude) was
skipped via `--no-teacher` to focus on the student comparison.

### Prompt 1: Binary tree balanced check (Python)

**Base student**: Produces code but with bugs — `nonlocal max_height` on a
parameter, confused logic mixing booleans and heights, incorrect comments
("if the node is None, it's not balanced" then returns True).

**Distilled student**: Cleaner structure with separate `get_height()` and
`validate_balance()` helpers, correct algorithm (compare heights, recurse both
sides), better docstring explaining what "balanced" means. Still has a bug
(`get_height` missing base case), but the structure and approach are notably
more Claude-like.

### Prompt 2: Connection pool (Python)

**Base student**: Confused implementation using a dict where it calls `.get()`
with no args, immediately stores None back, broken `set_connection` method.
Cut off mid-function.

**Distilled student**: Uses `threading.Event`, `contextmanager` import,
class-level lock, `acquire()`/`release()` API pattern, docstrings with usage
examples. Still imperfect (unnecessary `time.sleep`), but the design pattern
vocabulary is dramatically better.

### Prompt 3: SQL window functions

**Base student**: Completely wrong — uses bare `AVG()` without window function
(would fail in SQL), doesn't partition by department, invents nonexistent
column names (`emp_salary`), incorrect self-join suggestion.

**Distilled student**: Uses CTEs, `AVG(salary) OVER (PARTITION BY department_id)`,
proper JOIN, CASE expression. The SQL is more complex than needed (over-engineered)
but demonstrates actual window function knowledge the base model didn't have.

### Prompt 4: Java virtual threads

**Base student**: Almost entirely wrong — claims virtual threads are "managed by
the JVM rather than application code" (backwards), says `-Xms`/`-Xmx` controls
thread count (those control heap memory), confuses platform threads with OS
processes/PIDs.

**Distilled student**: Still inaccurate (confuses scheduling policies, mentions
LRU eviction which isn't relevant), but uses correct Java syntax, includes a code
example with `Thread.currentThread().getName()`, and at least frames it as a
concurrency concept rather than a memory management one. Improvement visible, but
Java knowledge transfer was weaker than Python/SQL.

## Key Observations

1. The distillation effect is **most visible in Python and SQL** — the distilled
   model picked up Claude's structural patterns (helper functions, docstrings,
   CTEs, window functions).

2. The distilled model's **vocabulary changed** — it uses terms like
   "acquire/release", "contextmanager", "CTE", "PARTITION BY" that the base
   model didn't.

3. The base model **confidently produces broken code** while the distilled model
   produces **more structured but still imperfect** code — a clear quality shift.

4. **Java knowledge transferred least well** — likely because with only ~100
   examples, there weren't enough Java-specific training pairs. At 16M exchanges,
   this gap would close substantially.

5. With only 104 training examples and 3 minutes of fine-tuning, a 0.5B parameter
   model shows measurable improvement. The Chinese labs used 16 million exchanges
   over months. Scale matters.
