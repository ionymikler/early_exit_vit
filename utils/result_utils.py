import json
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def plot_metrics(metrics):
    # Extract data for plotting
    exits = []
    counts = []
    accuracies = []
    confidences = []

    # Sort exits by their position
    for exit_key, stats in sorted(
        metrics["exit_statistics"].items(),
        key=lambda x: float("inf")
        if x[0] == "final"
        else int(x[0].split("_")[1])
        if "_" in x[0]
        else float("inf"),
    ):
        exits.append(
            "Final" if exit_key == "final" else f'Exit {exit_key.split("_")[1]}'
        )
        counts.append(stats["count"])
        accuracies.append(stats["accuracy"])
        confidences.append(stats["average_confidence"])

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Sample Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    x = range(len(exits))
    scatter = ax1.scatter(x, counts, c=counts, cmap="viridis", s=200)
    ax1.plot(x, counts, "--", color="gray", alpha=0.5)  # Add connecting lines
    ax1.set_title("Sample Distribution Across Exits")
    ax1.set_xlabel("Exit Point")
    ax1.set_ylabel("Number of Samples")
    ax1.set_xticks(x)
    ax1.set_xticklabels(exits, rotation=45)

    # Add percentage labels
    total_samples = metrics["total_samples"]
    for i, count in enumerate(counts):
        percentage = (count / total_samples) * 100
        ax1.text(x[i], count, f"{percentage:.1f}%", ha="center", va="bottom")

    # 2. Accuracy Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(exits, accuracies, color="lightgreen")
    ax2.set_title("Accuracy by Exit Point")
    ax2.set_xlabel("Exit Point")
    ax2.set_ylabel("Accuracy (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.axhline(
        y=metrics["overall_accuracy"],
        color="r",
        linestyle="--",
        label=f'Overall Accuracy ({metrics["overall_accuracy"]:.1f}%)',
    )

    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )
    ax2.legend()

    # 3. Confidence Levels
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(exits, confidences, color="salmon")
    ax3.set_title("Average Confidence by Exit Point")
    ax3.set_xlabel("Exit Point")
    ax3.set_ylabel("Confidence")
    ax3.tick_params(axis="x", rotation=45)

    # Add confidence values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # 4. Accuracy vs Confidence Scatter Plot
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(  # noqa F841
        confidences, accuracies, c=range(len(exits)), cmap="viridis", s=100
    )

    # Add labels for each point
    for i, exit_label in enumerate(exits):
        ax4.annotate(
            exit_label,
            (confidences[i], accuracies[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax4.set_title("Accuracy vs Confidence")
    ax4.set_xlabel("Confidence")
    ax4.set_ylabel("Accuracy (%)")

    # Add correlation coefficient
    correlation = np.corrcoef(confidences, accuracies)[0, 1]
    ax4.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.2f}",
        transform=ax4.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.suptitle("LGVIT Model Evaluation Metrics", fontsize=16, y=1.02)
    plt.tight_layout()

    return fig


def main():
    # Load and plot metrics
    metrics = load_metrics("results/evaluation_metrics_250219_121503.json")
    _ = plot_metrics(metrics)

    # Show the figure
    plt.show()

    # Save the figure after showing
    # save = input("Save figure? (y/n): ").lower().strip()
    # if save == "y":
    #     plt.savefig(
    #         "evaluation_metrics_visualization.png", bbox_inches="tight", dpi=300
    #     )

    plt.close()


if __name__ == "__main__":
    main()
