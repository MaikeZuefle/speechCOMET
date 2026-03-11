import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_score_distribution(input_path: str, output_path: str = "score_distribution.png"):
    # Load scores
    scores = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                scores.append(data if isinstance(data, (int, float)) else data["score"])

    scores = np.array(scores)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with KDE
    axes[0].hist(scores, bins=30, color="steelblue", edgecolor="white", alpha=0.85, density=True)
    kde = gaussian_kde(scores)
    x = np.linspace(scores.min(), scores.max(), 300)
    axes[0].plot(x, kde(x), color="tomato", linewidth=2, label="KDE")
    axes[0].set_title("Score Distribution", fontsize=14)
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(scores, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.7),
                    medianprops=dict(color="tomato", linewidth=2))
    axes[1].set_title("Score Box Plot", fontsize=14)
    axes[1].set_ylabel("Score")
    axes[1].set_xticks([])

    # Annotate stats
    stats = (f"n={len(scores)}  mean={scores.mean():.2f}  std={scores.std():.2f}\n"
             f"min={scores.min():.2f}  median={np.median(scores):.2f}  max={scores.max():.2f}")
    fig.text(0.5, 0.01, stats, ha="center", fontsize=10, color="gray")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot distribution of scores from a JSONL file.")
    parser.add_argument("input", help="Path to input .jsonl file")
    parser.add_argument("--output", default="score_distribution.png", help="Path to save the output plot (default: score_distribution.png)")
    args = parser.parse_args()

    plot_score_distribution(args.input, args.output)