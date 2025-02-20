import json
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from .logging_utils import get_logger_ready

logger = get_logger_ready(__name__)

# Color schemes for different backends
COLOR_SCHEMES = {
    "pytorch-cpu": {
        "primary": "#EE4C2C",  # PyTorch red
        "secondary": "#FFB991",  # Lighter shade
        "tertiary": "#FF7043",  # Medium shade
        "scatter": "Reds",  # Colormap for scatter plots
    },
    "pytorch-cuda": {
        "primary": "#76B900",  # NVIDIA green
        "secondary": "#B4E66E",  # Lighter shade
        "tertiary": "#98DC19",  # Medium shade
        "scatter": "Greens",  # Colormap for scatter plots
    },
    "onnxrt-cpu": {
        "primary": "#2878BD",  # ONNX blue
        "secondary": "#8CC7FF",  # Lighter shade
        "tertiary": "#5AA7FF",  # Medium shade
        "scatter": "Blues",  # Colormap for scatter plots
    },
}


def save_metrics(metrics, file_prefix: str):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    metrics_file = f"results/{file_prefix}_metrics_{timestamp}.json"
    latest_metrics_file = f"results/{file_prefix}_metrics_latest.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(latest_metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"\nMetrics saved to {metrics_file}")


def load_metrics(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def plot_metrics(metrics, title: str, color_scheme="pytorch-cpu"):
    # Get color scheme
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["pytorch-cpu"])

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
    scatter = ax1.scatter(x, counts, c=counts, cmap=colors["scatter"], s=200)  # noqa F841
    ax1.plot(x, counts, "--", color=colors["tertiary"], alpha=0.5)
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
    bars = ax2.bar(exits, accuracies, color=colors["primary"])
    ax2.set_title("Accuracy by Exit Point")
    ax2.set_xlabel("Exit Point")
    ax2.set_ylabel("Accuracy (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.axhline(
        y=metrics["overall_accuracy"],
        color=colors["tertiary"],
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
    bars = ax3.bar(exits, confidences, color=colors["secondary"])
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
        confidences, accuracies, c=range(len(exits)), cmap=colors["scatter"], s=100
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

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    return fig


def choose_metrics_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        initialdir="results/",
        title="Select metrics file",
        filetypes=(("JSON files", "*.json"), ("all files", "*.*")),
    )
    return file_path if file_path else None


def choose_color_scheme():
    import tkinter as tk
    from tkinter import ttk

    def on_select():
        nonlocal selected_scheme
        selected_scheme = combo.get()
        root.quit()

    root = tk.Tk()
    root.title("Select Color Scheme")

    selected_scheme = None

    # Center window
    window_width = 300
    window_height = 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

    # Add label
    label = ttk.Label(root, text="Select visualization color scheme:")
    label.pack(pady=10)

    # Add combobox
    combo = ttk.Combobox(root, values=list(COLOR_SCHEMES.keys()))
    combo.set("pytorch-cpu")  # default value
    combo.pack(pady=10)

    # Add button
    button = ttk.Button(root, text="OK", command=on_select)
    button.pack(pady=10)

    root.mainloop()
    root.destroy()

    return selected_scheme or "pytorch-cpu"
