"""
Generate visual charts for the distillation demo video.

Creates PNG images that can be displayed on screen during recording:
  - Training loss and accuracy curves
  - Model comparison scorecard (radar chart)
  - Cost scaling projection

Usage:
    python generate_visuals.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "font.size": 14,
    "axes.titlesize": 18,
    "figure.titlesize": 22,
})

OUTPUT_DIR = "./visuals"


def ensure_output_dir():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_curves():
    """Training loss and accuracy over the 5-epoch run."""

    # Data from the actual training logs (1.5B model, 5 epochs)
    steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    loss = [0.900, 0.869, 0.787, 0.806, 0.804, 0.757, 0.781, 0.800, 0.684, 0.809,
            None, None, None]
    accuracy = [78.1, 79.5, 80.0, 79.5, 80.5, 80.2, 79.7, 82.2, 79.6,
                None, None, None, None]
    epochs = [0.385, 0.769, 1.154, 1.538, 1.923, 2.308, 2.692, 3.077,
              3.462, 3.846, 4.231, 4.615, 5.0]

    # Filter out None values
    loss_data = [(e, l) for e, l in zip(epochs, loss) if l is not None]
    acc_data = [(e, a) for e, a in zip(epochs, accuracy) if a is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training Progress: Qwen 1.5B on Claude's Outputs", fontweight="bold")

    # Loss curve
    ex, ly = zip(*loss_data)
    ax1.plot(ex, ly, "o-", color="#00d4ff", linewidth=2.5, markersize=8)
    ax1.fill_between(ex, ly, alpha=0.15, color="#00d4ff")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5.2)

    # Accuracy curve
    ex2, ay = zip(*acc_data)
    ax2.plot(ex2, ay, "s-", color="#ff6b6b", linewidth=2.5, markersize=8)
    ax2.fill_between(ex2, ay, alpha=0.15, color="#ff6b6b")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Token Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5.2)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_scorecard():
    """Radar chart comparing the three models across quality dimensions."""

    categories = [
        "Code\nCorrectness",
        "Structure &\nDesign",
        "Documentation\nQuality",
        "Error\nHandling",
        "Completeness",
    ]

    # Scores out of 10 (averaged across the 4 test prompts)
    teacher =   [9.5, 9.5, 9.0, 9.0, 8.5]  # truncated by token limit
    base =      [5.0, 4.0, 4.5, 2.5, 5.0]
    distilled = [6.0, 7.5, 8.0, 7.0, 6.5]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the polygon
    teacher += teacher[:1]
    base += base[:1]
    distilled += distilled[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.suptitle("Model Quality Comparison", fontweight="bold", y=0.98)

    ax.set_facecolor("#16213e")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw each model
    ax.plot(angles, teacher, "o-", linewidth=2.5, label="Teacher (Claude)",
            color="#ffd93d", markersize=8)
    ax.fill(angles, teacher, alpha=0.1, color="#ffd93d")

    ax.plot(angles, base, "s-", linewidth=2.5, label="Base Student (1.5B)",
            color="#ff6b6b", markersize=8)
    ax.fill(angles, base, alpha=0.1, color="#ff6b6b")

    ax.plot(angles, distilled, "D-", linewidth=2.5, label="Distilled Student (1.5B)",
            color="#00d4ff", markersize=8)
    ax.fill(angles, distilled, alpha=0.15, color="#00d4ff")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=13)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], size=10)
    ax.yaxis.grid(True, color="#2a2a4a", alpha=0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=13,
              facecolor="#1a1a2e", edgecolor="#e0e0e0")

    path = f"{OUTPUT_DIR}/model_scorecard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cost_scaling():
    """Bar chart showing cost at different scales."""

    labels = [
        "Our demo\n(104 prompts)",
        "Small campaign\n(10K prompts)",
        "Medium campaign\n(1M prompts)",
        "DeepSeek/MiniMax\n(16M prompts)",
    ]
    costs = [1.61, 155, 15_500, 248_000]
    colors = ["#00d4ff", "#4ecdc4", "#ffd93d", "#ff6b6b"]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Distillation Cost Scaling (Estimated)", fontweight="bold")

    bars = ax.bar(labels, costs, color=colors, edgecolor="#e0e0e0", linewidth=0.5,
                  width=0.6)

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        if cost < 100:
            label = f"${cost:.2f}"
        else:
            label = f"${cost:,.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                label, ha="center", va="bottom", fontweight="bold", fontsize=14)

    ax.set_ylabel("Estimated Cost (USD)")
    ax.set_yscale("log")
    ax.set_ylim(1, 500_000)
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotation
    ax.annotate("All of this is revenue\nfor Anthropic",
                xy=(3, 248_000), xytext=(1.5, 300_000),
                fontsize=13, fontstyle="italic", color="#ff6b6b",
                arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=2))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/cost_scaling.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_lora_diagram():
    """Illustrate how LoRA works: frozen W + trainable low-rank A and B."""

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("LoRA: Low-Rank Adaptation", fontweight="bold")

    W_COLOR = "#4a4a6a"  # same neutral color for W on both sides

    # --- Helper to draw a labeled matrix block ---
    def draw_matrix(x, y, w, h, label, color, sublabel=None):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#e0e0e0",
                              linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=16, fontweight="bold", color="#fff")
        if sublabel:
            ax.text(x + w / 2, y - 0.3, sublabel, ha="center", va="top",
                    fontsize=11, color="#aaa", style="italic")

    # Layout constants
    mid_y = 3.5

    # --- Left side: Standard fine-tuning ---
    ax.text(3.0, 7.7, "Standard Fine-Tuning", ha="center", fontsize=15,
            fontweight="bold", color="#ff6b6b")
    ax.text(3.0, 7.1, "Update ALL of W directly", ha="center", fontsize=12,
            color="#aaa")
    draw_matrix(1.0, mid_y, 4.0, 3.0, "W\n(d × d)", W_COLOR,
                sublabel="1.5 billion params — all trainable")

    # Red border to indicate "trainable"
    highlight = plt.Rectangle((1.0, mid_y), 4.0, 3.0, facecolor="none",
                               edgecolor="#ff6b6b", linewidth=3, linestyle="--")
    ax.add_patch(highlight)

    # --- Right side: LoRA ---
    ax.text(11.0, 7.7, "LoRA Fine-Tuning", ha="center", fontsize=15,
            fontweight="bold", color="#00d4ff")
    ax.text(11.0, 7.1, "Freeze W, train only B and A", ha="center",
            fontsize=12, color="#aaa")

    # Frozen W (same color as left side)
    draw_matrix(7.5, mid_y, 3.0, 3.0, "W\n(d × d)", W_COLOR,
                sublabel="1.5B params — frozen")
    ax.text(9.0, mid_y + 3.15, "FROZEN", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#aaa")

    # Plus sign
    ax.text(11.0, mid_y + 1.5, "+", ha="center", va="center",
            fontsize=28, fontweight="bold", color="#e0e0e0")

    # B matrix (d × r) — tall and narrow
    draw_matrix(11.8, mid_y + 0.4, 0.8, 2.2, "B", "#0288d1",
                sublabel="d × r")

    # Multiplication dot
    ax.text(13.0, mid_y + 1.5, "×", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#e0e0e0")

    # A matrix (r × d) — short and wide
    draw_matrix(13.4, mid_y + 0.8, 2.2, 0.8, "A", "#0288d1",
                sublabel="r × d")

    # Cyan border around B×A to indicate "trainable"
    trainable_box = plt.Rectangle((11.6, mid_y + 0.2), 4.2, 2.6, facecolor="none",
                                   edgecolor="#00d4ff", linewidth=3, linestyle="--")
    ax.add_patch(trainable_box)
    ax.text(13.7, mid_y + 2.95, "TRAINABLE", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#00d4ff")

    # Annotation: rank — position to the right to avoid overlap
    ax.annotate("r = 16", xy=(12.2, mid_y + 0.3), xytext=(11.2, mid_y - 1.0),
                fontsize=13, fontweight="bold", color="#ffd93d",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#ffd93d", lw=1.5))

    # --- Forward pass equation ---
    ax.text(8.5, 1.5, "Forward pass:   output  =  W·x  +  B·A·x",
            ha="center", fontsize=15, fontweight="bold", color="#e0e0e0",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a4a",
                      edgecolor="#e0e0e0", linewidth=1.5))

    # --- Caption explaining B×A ---
    ax.text(8.5, 0.5,
            "B × A approximates the weight update $\\Delta$W "
            "that standard fine-tuning would have made",
            ha="center", fontsize=12, color="#aaa", style="italic")

    # --- Bottom: parameter comparison ---
    ax.text(3.0, 2.5, "100% of parameters trained", ha="center",
            fontsize=13, color="#ff6b6b", fontweight="bold")
    ax.text(13.7, 2.5, "0.1% of parameters trained", ha="center",
            fontsize=13, color="#00d4ff", fontweight="bold")
    ax.text(13.7, 2.0, "(1.7 million out of 1.5 billion)", ha="center",
            fontsize=11, color="#aaa")

    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/lora_diagram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_prompt_comparison():
    """Grouped bar chart showing base vs distilled scores per prompt."""

    prompts = [
        "Binary Tree\n(Python)",
        "Connection Pool\n(Python)",
        "Window Functions\n(SQL)",
        "TTL Cache\n(Python)",
    ]

    base_scores =      [6.0, 4.0, 5.5, 4.5]
    distilled_scores = [7.5, 8.0, 7.5, 7.0]

    x = np.arange(len(prompts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Base vs Distilled Student: Per-Prompt Quality", fontweight="bold")

    bars1 = ax.bar(x - width/2, base_scores, width, label="Base Student",
                   color="#ff6b6b", edgecolor="#e0e0e0", linewidth=0.5)
    bars2 = ax.bar(x + width/2, distilled_scores, width, label="Distilled Student",
                   color="#00d4ff", edgecolor="#e0e0e0", linewidth=0.5)

    # Add improvement arrows
    for i, (b, d) in enumerate(zip(base_scores, distilled_scores)):
        improvement = d - b
        ax.annotate(f"+{improvement:.1f}",
                    xy=(i + width/2, d),
                    xytext=(i + width/2, d + 0.5),
                    ha="center", fontweight="bold", fontsize=13, color="#4ecdc4")

    ax.set_ylabel("Quality Score (0-10)")
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=12)
    ax.set_ylim(0, 10.5)
    ax.legend(fontsize=13, facecolor="#1a1a2e", edgecolor="#e0e0e0")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/prompt_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    ensure_output_dir()
    print("Generating visuals for the video...\n")

    plot_training_curves()
    plot_scorecard()
    plot_cost_scaling()
    plot_prompt_comparison()
    plot_lora_diagram()

    print(f"\nAll visuals saved to {OUTPUT_DIR}/")
    print("Files:")
    print("  training_curves.png   — Loss & accuracy over 5 epochs")
    print("  model_scorecard.png   — Radar chart: Teacher vs Base vs Distilled")
    print("  cost_scaling.png      — What this costs at scale")
    print("  prompt_comparison.png — Base vs Distilled per prompt")
    print("  lora_diagram.png     — How LoRA works (frozen W + small A, B)")
