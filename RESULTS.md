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

- **Student model**: `Qwen/Qwen2.5-1.5B-Instruct` (1.5B params, LoRA adapters)
- **Time**: ~11 minutes on M4 Max (MPS)
- **Epochs**: 5
- **Loss**: 0.900 → 0.809
- **Accuracy**: 78.1% → 79.6% (best batch: 82.2% at epoch 4.6)
- **Output**: `distilled-student/` directory

## Step 3: Comparison (`03_compare_models.py`)

Four test prompts comparing all three models: Teacher (Claude), Base Student
(before distillation), and Distilled Student (after training on Claude's outputs).

### Prompt 1: Binary tree balanced check (Python)

**Teacher (Claude)**: Uses `@dataclass` for TreeNode, `Optional["TreeNode"]` forward
refs, a `check_height` helper returning -1 for early termination on imbalance.
Includes docstrings with `Examples:` section and doctests. Production-quality code.

**Base student**: Correct overall structure — defines TreeNode, `is_balanced`, and a
`check_balance` helper. Uses -2 as sentinel for imbalance (unusual but workable).
Good docstring. Cut off before completing the balance check logic, but the approach
is reasonable.

**Distilled student**: Adds a `height` `@property` to TreeNode, makes `is_balanced`
a method on the class (OOP style vs Claude's functional style). Includes detailed
docstrings with Attributes/Methods sections. More elaborate class design — the
structure and documentation style clearly echo Claude's patterns.

### Prompt 2: Connection pool (Python)

**Teacher (Claude)**: Full production design — `@dataclass` Connection wrapper with
metadata (created_at, last_used_at, use_count), context manager protocol, Queue-based
pool, logging, connection validation, pool statistics. Imports threading, time, logging,
contextmanager, dataclass, Queue.

**Base student**: Basic but functional — uses `threading.Lock()`, list-based pool,
`get_connection`/`put_connection` API. Logic bug (pops from empty pool on `get`).
Includes ThreadPoolExecutor usage example. Simple but shallow.

**Distilled student**: Dramatically more sophisticated — custom exception hierarchy
(ConnectionPoolError, ConnectionNotFoundError, FullConnectionPoolError,
AlreadyConnectedError, etc.), Connection class with attributes, typed imports. The
error handling vocabulary and class design clearly mirror Claude's style. Over-engineered
for the task, but the leap in design pattern sophistication from base is striking.

### Prompt 3: SQL window functions

**Teacher (Claude)**: Perfect SQL — uses `AVG(salary) OVER (PARTITION BY department_id)`
in a subquery, filters with `WHERE salary > dept_avg_salary`, includes `ROUND()`,
percentage above average calculation, `JOIN departments`, `ORDER BY`. Provides sample
data tables and expected output.

**Base student**: Correct use of `AVG(salary) OVER (PARTITION BY department)` — gets
the window function right. But stops there: no filtering for above-average, no
subquery/CTE, just selects the average alongside each row. Mentions comparing but
doesn't actually implement the filter. Incomplete solution.

**Distilled student**: Uses CTEs (`DepartmentAverages`, `EmployeeSalariesAboveAvg`),
`AVG(salary) OVER (PARTITION BY department_id)`, `ROW_NUMBER()`, JOINs across
multiple tables, CASE expression for improvement status. Over-complex (ROW_NUMBER
isn't needed here), but demonstrates CTE vocabulary, multi-table joins, and
structured query organization the base model didn't attempt. Includes section
headers and explanation breakdown mimicking Claude's format.

### Prompt 4: TTL cache decorator (Python)

**Teacher (Claude)**: Full `CacheEntry` dataclass with `is_expired` and
`ttl_remaining` properties, a `TTLCache` class with thread-safe RLock, LRU eviction
via OrderedDict, maxsize support, and hit/miss statistics. Clean separation of
concerns.

**Base student**: Correct basic idea — nested decorator with `_cache` dict, key
generation from function name + args. But the TTL logic is broken: measures elapsed
time *within a single call* (start_time to end_time) instead of tracking when the
entry was cached. Raises an exception on stale cache hit instead of re-computing.
Functional structure but flawed implementation.

**Distilled student**: Builds a full cache infrastructure — `CacheError` exception,
`get_cache()` function with module-level state, warning classes (CacheMissWarning,
CacheHitWarning, CacheFullWarning), and a `Cache` class with `set`/`get`/`delete`/
`exists`/`flush` API and O(1) complexity claim. Over-architected, but the vocabulary
(custom exceptions, warning classes, cache API design) clearly reflects Claude's
influence. Far more sophisticated design than the base model's simple dict approach.

## Key Observations

1. **The 1.5B model shows a much clearer distillation effect than 0.5B.** The base
   1.5B model already writes reasonable code, but the distilled version's *style*
   and *design vocabulary* visibly shift toward Claude's patterns.

2. **The distilled model consistently over-engineers.** Custom exception hierarchies,
   warning classes, dataclass wrappers — it picked up Claude's tendency toward
   production-grade structure, even when the prompt doesn't call for it. This is
   actually the most visible evidence of distillation: the base model writes simple
   code, the distilled model writes *Claude-shaped* code.

3. **Documentation style transfer is the clearest signal.** The distilled model
   adds section headers (## Overview, ### Requirements), structured explanations,
   and detailed docstrings that the base model doesn't produce. This formatting
   is pure Claude.

4. **The SQL prompt shows the most dramatic improvement.** The base model gets the
   window function right but doesn't complete the solution. The distilled model
   builds CTEs, multi-table JOINs, and structured query organization — techniques
   present in the training data from Claude.

5. **Scale matters.** With 104 training examples and 11 minutes of fine-tuning,
   the effect is visible but modest. The Chinese labs used 16 million exchanges
   over months. At that scale, the capability transfer would be substantial.

---

## Previous Run (0.5B model, 3 epochs)

An earlier test run used `Qwen/Qwen2.5-0.5B-Instruct` with 3 epochs:
- Training time: ~3 minutes, Loss: 1.289 → 0.991, Accuracy: 72.7% → 76.0%
- The distillation effect was visible but weaker — the 0.5B model has less
  capacity to absorb Claude's patterns. The 1.5B model was a clear improvement
  for demo purposes.
