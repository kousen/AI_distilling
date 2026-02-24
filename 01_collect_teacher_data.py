"""
Step 1: Collect Training Data from the Teacher Model (Claude)
=============================================================

This is the conceptual equivalent of what Anthropic accused DeepSeek,
MiniMax, and Moonshot of doing — sending carefully crafted prompts to
Claude and collecting the responses as training data.

They did it 16 million times across 24,000 fraudulent accounts.
We're doing it ~100 times with one honest account (32 base prompts
expanded to ~100 via programmatic variations). Same technique,
very different scale.

Usage:
    export ANTHROPIC_API_KEY=your-key-here
    python 01_collect_teacher_data.py
"""

import json
import time
import anthropic

# ── Configuration ────────────────────────────────────────────────
TEACHER_MODEL = "claude-sonnet-4-6"
OUTPUT_FILE = "teacher_data.jsonl"
MAX_TOKENS = 1024

# ── The Prompts ──────────────────────────────────────────────────
# Anthropic said the Chinese labs "targeted Claude's most differentiated
# capabilities: agentic reasoning, tool use, and coding."
#
# In a real distillation campaign, you'd generate thousands of these
# programmatically. MiniMax allegedly used 13 million exchanges focused
# on agentic coding alone.

CODING_PROMPTS = [
    # Data structures & algorithms
    "Write a Python function to find the longest common subsequence of two strings. Include type hints and a brief explanation of the approach.",
    "Implement a trie (prefix tree) in Python with insert, search, and starts_with methods.",
    "Write a Python function that solves the N-Queens problem using backtracking. Return all valid board configurations.",
    "Implement Dijkstra's shortest path algorithm in Python using a min-heap.",
    "Write a Python class for a LRU (Least Recently Used) cache with O(1) get and put operations.",
    "Implement a red-black tree insertion in Python with proper rebalancing.",
    "Write a Python function to find all strongly connected components in a directed graph using Tarjan's algorithm.",
    "Implement a segment tree in Python that supports range sum queries and point updates.",

    # Practical coding tasks
    "Write a Python decorator that retries a function up to N times with exponential backoff if it raises an exception.",
    "Write a Python context manager that temporarily redirects stdout to a file and restores it on exit.",
    "Implement a simple thread pool in Python using only the threading and queue modules.",
    "Write a Python function that parses a cron expression and returns the next N scheduled run times.",
    "Implement a rate limiter using the token bucket algorithm in Python. It should be thread-safe.",
    "Write a Python function that takes a nested JSON object and flattens it into dot-notation keys.",
    "Implement a simple pub/sub event system in Python with subscribe, unsubscribe, and publish methods.",
    "Write a Python function to detect cycles in a dependency graph and report which packages are involved.",

    # Systems / DevOps
    "Write a bash script that monitors disk usage across all mounted partitions and sends an alert if any exceeds 90%.",
    "Write a Python script that watches a directory for new CSV files, validates their schema, and loads them into a SQLite database.",
    "Write a Dockerfile for a Python Flask app that uses multi-stage builds to minimize image size.",
    "Write a Python script that performs health checks on a list of URLs concurrently using asyncio and aiohttp.",

    # SQL
    "Write a SQL query to find the second highest salary in each department. Use window functions.",
    "Write a SQL query that identifies customers who made purchases in three consecutive months.",
    "Write a recursive CTE in SQL to generate a hierarchical org chart from an employees table with a manager_id column.",
    "Write SQL to calculate a 7-day rolling average of daily sales, handling gaps in dates.",

    # Java / JVM
    "Write a Java record that implements a simple immutable linked list with head, tail, map, and filter methods using sealed interfaces.",
    "Implement the Observer pattern in Kotlin using flows and coroutines.",
    "Write a Spring Boot REST controller with proper exception handling, validation, and pagination for a Book entity.",
    "Show how to implement a retry mechanism in Java using virtual threads (Project Loom) without any external libraries.",

    # Explanations (testing reasoning extraction)
    "Explain the difference between composition and inheritance in object-oriented design. Give concrete examples in Java showing when each is appropriate.",
    "Explain how garbage collection works in the JVM. Cover the generational hypothesis, G1 vs ZGC, and when you'd choose one over the other.",
    "Explain the CAP theorem with concrete examples of real databases that make different tradeoffs.",
    "Explain how Python's GIL works and why it matters for concurrent programming. Show the difference between threading and multiprocessing with benchmarks.",
]

# ── Prompt Expansion ─────────────────────────────────────────
# 32 base prompts isn't enough to reliably shift the student model's
# behavior. In a real campaign you'd generate thousands programmatically.
# We'll apply modifiers to subsets of the base prompts to produce
# variations that elicit genuinely different responses from the teacher.

# Each modifier is (suffix_text, applicable_indices) where indices
# refer to positions in CODING_PROMPTS. None means "apply to all."
PROMPT_MODIFIERS = [
    # Variation: add robustness
    (
        " Add comprehensive error handling and edge case checks.",
        list(range(0, 16)),  # data structures + practical tasks
    ),
    # Variation: testing
    (
        " Include unit tests using pytest that cover normal cases and edge cases.",
        list(range(0, 16)),  # data structures + practical tasks
    ),
    # Variation: complexity analysis
    (
        " Analyze the time and space complexity of your solution.",
        list(range(0, 8)),  # data structures & algorithms
    ),
    # Variation: alternative approach
    (
        " Now implement an alternative approach and compare the tradeoffs.",
        list(range(0, 8)) + list(range(20, 24)),  # algorithms + SQL
    ),
    # Variation: step-by-step reasoning
    (
        " Walk through your reasoning step by step before writing the code.",
        list(range(8, 20)),  # practical + systems
    ),
    # Variation: production readiness
    (
        " Make this production-ready with logging, type hints, and docstrings.",
        list(range(8, 16)),  # practical tasks
    ),
]


def expand_prompts(base_prompts):
    """Apply modifiers to base prompts to create a larger training set.

    This is a simplified version of what a real distillation campaign
    does at scale — programmatically generating prompt variations to
    maximize coverage of the target capability.
    """
    expanded = list(base_prompts)  # start with all originals

    for suffix, indices in PROMPT_MODIFIERS:
        for i in indices:
            if i < len(base_prompts):
                expanded.append(base_prompts[i] + suffix)

    # Deduplicate (shouldn't happen, but just in case)
    seen = set()
    unique = []
    for p in expanded:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


ALL_PROMPTS = expand_prompts(CODING_PROMPTS)


def collect_teacher_responses():
    """
    Send prompts to Claude and collect responses.
    This is the core of what "distillation" means in the black-box sense.
    """
    client = anthropic.Anthropic()
    training_data = []
    errors = []

    print(f"Collecting responses from {TEACHER_MODEL}...")
    print(f"Total prompts: {len(ALL_PROMPTS)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)

    for i, prompt in enumerate(ALL_PROMPTS):
        try:
            response = client.messages.create(
                model=TEACHER_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )

            completion = response.content[0].text
            entry = {
                "prompt": prompt,
                "completion": completion,
                "model": TEACHER_MODEL,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            training_data.append(entry)

            print(f"[{i+1}/{len(ALL_PROMPTS)}] ✓ {prompt[:60]}...")
            print(f"         tokens: {entry['input_tokens']} in / {entry['output_tokens']} out")

            # Be polite to the API (and your wallet)
            time.sleep(0.5)

        except Exception as e:
            print(f"[{i+1}/{len(ALL_PROMPTS)}] ✗ Error: {e}")
            errors.append({"prompt": prompt, "error": str(e)})

    # Write training data
    with open(OUTPUT_FILE, "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")

    # Summary
    total_input = sum(d["input_tokens"] for d in training_data)
    total_output = sum(d["output_tokens"] for d in training_data)

    print("\n" + "=" * 60)
    print(f"COLLECTION COMPLETE")
    print(f"  Successful: {len(training_data)}/{len(ALL_PROMPTS)}")
    print(f"  Errors:     {len(errors)}")
    print(f"  Total tokens: {total_input:,} input + {total_output:,} output")
    print(f"  Saved to: {OUTPUT_FILE}")
    print()

    # ── The irony ────────────────────────────────────────────────
    # Anthropic charges per token. Every one of those 16 million
    # exchanges PAID Anthropic for the privilege of being distilled.
    # Let's estimate what our tiny demo cost vs what the Chinese
    # labs' campaigns might have cost.
    sonnet_input_cost = 3.00 / 1_000_000   # $3 per 1M input tokens
    sonnet_output_cost = 15.00 / 1_000_000  # $15 per 1M output tokens
    our_cost = (total_input * sonnet_input_cost) + (total_output * sonnet_output_cost)
    print(f"  Estimated cost of our demo:  ${our_cost:.2f}")
    print(f"  (The Chinese labs did this ~500,000x more)")
    print(f"  Estimated cost at 16M exchanges: ${our_cost * (16_000_000 / len(ALL_PROMPTS)):,.0f}")
    print(f"  That's revenue for Anthropic. From the 'attack.'")

    return training_data


if __name__ == "__main__":
    collect_teacher_responses()
