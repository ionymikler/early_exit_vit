import os
import matplotlib.pyplot as plt
from tkinter import messagebox
import tkinter as tk
from utils import result_utils, logging_utils

logger = logging_utils.get_logger_ready(__name__)


def save_figure(fig, metrics_path):
    """Save figure as PNG in the same directory as the metrics file."""
    # Get base path and create new filename
    base_path = os.path.splitext(metrics_path)[0]
    save_path = f"{base_path}_plot.png"

    # Save the figure
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Figure saved to: {save_path}")


def main():
    file_path = result_utils.choose_metrics_file()
    if file_path:
        metrics = result_utils.load_metrics(file_path)
        color_scheme = result_utils.choose_color_scheme()
        title = f"LGVIT Model Evaluation Metrics - {os.path.basename(file_path).removesuffix('.json')}"

        fig = result_utils.plot_metrics(metrics, title, color_scheme)
        plt.show()

        # Simple yes/no prompt to save
        root = tk.Tk()
        root.withdraw()
        if messagebox.askyesno("Save Figure", "Would you like to save the figure?"):
            save_figure(fig, file_path)

        plt.close()
    else:
        logger.warning("No file selected")


if __name__ == "__main__":
    main()
