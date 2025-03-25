#!/usr/bin/env python
# Enhanced CIFAR-100 Dataset Viewer
# Displays random images from a specific class in the CIFAR-100 dataset
# With options to save individual images

import argparse
import random
import matplotlib.pyplot as plt
from datasets import load_dataset
from typing import List, Tuple
import os
from PIL import Image
import math


def get_cifar100_dataset():
    """Load CIFAR-100 dataset from Huggingface datasets."""
    print("Loading CIFAR-100 dataset...")
    dataset = load_dataset(
        path="cifar100",
        cache_dir="./tmp/data/cifar100",
    )
    print(f"Dataset splits available: {dataset.keys()}")
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    return dataset


def get_class_names(dataset) -> List[str]:
    """Get all class names from the dataset."""
    return dataset["train"].features["fine_label"].names


def get_images_by_label(
    dataset, label_idx: int, split: str = "train", count: int = 15
) -> List[Tuple]:
    """
    Get random images from a specific class.

    Args:
        dataset: The CIFAR-100 dataset
        label_idx: The class index to retrieve images for
        split: Dataset split to use ('train' or 'test')
        count: Number of images to retrieve

    Returns:
        List of tuples containing (image, label_name, image_idx)
    """
    # Filter dataset to only include examples of the specified class
    filtered_dataset = dataset[split].filter(
        lambda example: example["fine_label"] == label_idx
    )

    if len(filtered_dataset) == 0:
        print(f"No images found for class index {label_idx}")
        return []

    # Sample random indices
    sample_size = min(count, len(filtered_dataset))
    indices = random.sample(range(len(filtered_dataset)), sample_size)

    # Get class name
    label_name = dataset[split].features["fine_label"].names[label_idx]

    # Get image data
    result = []
    for idx in indices:
        image = filtered_dataset[idx]["img"]
        # Use the index as a reference ID since 'id' field doesn't exist
        result.append((image, label_name, idx))

    return result


def save_individual_images(images: List[Tuple], save_dir: str):
    """
    Save individual images to the specified directory.

    Args:
        images: List of tuples containing (image, label_name, image_idx)
        save_dir: Directory to save images to
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    saved_paths = []
    for img, label_name, img_idx in images:
        # Create a clean label for the filename (remove spaces and special characters)
        clean_label = label_name.replace(" ", "_").lower()

        # Create filename: label_id.png
        filename = f"{clean_label}_{img_idx}.png"
        save_path = os.path.join(save_dir, filename)

        # Convert to PIL Image and save
        # Check if img is already a PIL Image or needs conversion
        if isinstance(img, Image.Image):
            pil_img = img
        else:
            # Handle numpy array or other format
            pil_img = Image.fromarray(img)

        pil_img.save(save_path)
        saved_paths.append(save_path)
        print(f"Saved: {save_path}")

    return saved_paths


def display_images(
    images: List[Tuple],
    rows: int = None,
    cols: int = 5,
    title: str = None,
    save_figure_path: str = None,
    save_individual: bool = False,
    save_dir: str = None,
):
    """
    Display images in a grid.

    Args:
        images: List of tuples containing (image, label_name, image_idx)
        rows: Number of rows in the grid (calculated automatically if None)
        cols: Number of columns in the grid (default is 5)
        title: Optional title for the figure
        save_figure_path: Path to save the figure (if None, figure is not saved)
        save_individual: Whether to save individual images
        save_dir: Directory to save individual images to (if None, user will be prompted)
    """
    # Calculate number of rows based on number of images and columns
    if rows is None:
        rows = math.ceil(len(images) / cols)

    # Create a figure with the appropriate size
    # Scale figure size based on number of images
    fig_width = min(15, cols * 3)  # Max width of 15, otherwise 3 units per column
    fig_height = min(12, rows * 3)  # Max height of 12, otherwise 3 units per row

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Increase font size for title
    if title:
        fig.suptitle(title, fontsize=20)  # Increased from default

    # Handle case where there's only one row or column
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    for i, (img, label_name, img_idx) in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img)
            # Increase font size for labels
            axes[i].set_title(f"Label: {label_name}\nID: {img_idx}", fontsize=14)
            axes[i].axis("off")

    # Hide any unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    # Apply tight layout first

    # Then adjust the top margin to make room for the title
    # Use a larger margin when title is present to prevent overlap
    plt.subplots_adjust(top=0.85)  # Increased space for title (was 0.9)
    # if title:
    # else:
    #     plt.subplots_adjust(top=0.95)

    plt.tight_layout()

    # Save whole figure if save_figure_path is provided
    if save_figure_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_figure_path), exist_ok=True)
        plt.savefig(save_figure_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_figure_path}")

    # Save individual images if requested
    if save_individual:
        if save_dir is None:
            save_dir = input("Enter directory to save individual images: ")
        save_individual_images(images, save_dir)

    plt.show()


def interactive_mode():
    """Run the script in interactive mode, allowing the user to view multiple classes."""
    dataset = get_cifar100_dataset()
    class_names = get_class_names(dataset)

    # Print all available classes with their indices
    print("\nAvailable classes:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    while True:
        try:
            # Get user input
            label_input = input("\nEnter class index (0-99) or name (or 'q' to quit): ")

            if label_input.lower() == "q":
                break

            # Try to parse as index first
            try:
                label_idx = int(label_input)
                if label_idx < 0 or label_idx >= 100:
                    print("Invalid index. Please enter a number between 0 and 99.")
                    continue
            except ValueError:
                # Try to find by name
                label_input = label_input.lower()
                matches = [
                    i
                    for i, name in enumerate(class_names)
                    if label_input in name.lower()
                ]

                if not matches:
                    print(f"No class names contain '{label_input}'. Please try again.")
                    continue

                if len(matches) > 1:
                    print("Multiple matches found:")
                    for idx in matches:
                        print(f"{idx}: {class_names[idx]}")
                    continue

                label_idx = matches[0]

            # Get images from both train and test sets
            split = (
                input("Choose split (train/test) [default: test]: ").lower() or "test"
            )
            if split not in ["train", "test"]:
                print("Invalid split. Using 'test'.")
                split = "test"

            # Get number of images to display
            try:
                count = int(input("Number of images to display [default: 15]: ") or 15)
            except ValueError:
                print("Invalid number. Using default (15).")
                count = 15

            # Ask if user wants to save the figure
            save_fig = (
                input("Save the complete figure? (y/n) [default: n]: ").lower() == "y"
            )
            save_fig_path = None

            if save_fig:
                # Create path for saving the figure
                class_name = class_names[label_idx].replace(" ", "_").lower()
                save_dir = "results/cifar100_viewer"
                save_fig_path = f"{save_dir}/{class_name}_sample.png"

            # Ask if user wants to save individual images
            save_individual = (
                input("Save individual images? (y/n) [default: n]: ").lower() == "y"
            )
            save_individual_dir = None

            if save_individual:
                save_individual_dir = input(
                    "Enter directory to save individual images [default: results/cifar100_images]: "
                )
                if not save_individual_dir:
                    save_individual_dir = "results/cifar100_images"

            # Get and display images
            images = get_images_by_label(dataset, label_idx, split, count)

            if images:
                # Always use 5 columns, rows will be calculated automatically
                cols = 5

                display_images(
                    images,
                    rows=None,  # Auto-calculate rows
                    cols=cols,
                    title=f"Class {label_idx}: {class_names[label_idx]} ({split} split)",
                    save_figure_path=save_fig_path,
                    save_individual=save_individual,
                    save_dir=save_individual_dir,
                )

        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced CIFAR-100 Dataset Viewer")
    parser.add_argument("--label", type=int, help="Class label index (0-99)")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Dataset split to use (train or test)",
    )
    parser.add_argument(
        "--count", type=int, default=15, help="Number of images to display"
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="results/cifar100_viewer",
        help="Directory to save figures (default: results/cifar100_viewer)",
    )
    parser.add_argument(
        "--save-figure",
        action="store_true",
        help="Save the complete figure",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual images",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save individual images",
    )

    args = parser.parse_args()

    if args.label is not None:
        # Command-line mode
        dataset = get_cifar100_dataset()
        class_names = get_class_names(dataset)

        if args.label < 0 or args.label >= 100:
            print("Error: Label index must be between 0 and 99.")
            return

        images = get_images_by_label(dataset, args.label, args.split, args.count)

        if images:
            # Always use 5 columns
            cols = 5

            # Create path for saving the figure if needed
            save_fig_path = None
            if args.save_figure:
                class_name = class_names[args.label].replace(" ", "_").lower()
                save_fig_path = f"{args.directory}/{class_name}_sample.png"

            display_images(
                images,
                rows=None,  # Auto-calculate rows
                cols=cols,
                title=f"Class {args.label}: {class_names[args.label]} ({args.split} split)",
                save_figure_path=save_fig_path,
                save_individual=args.save_individual,
                save_dir=args.save_dir,
            )
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    # Import numpy here to avoid importing it when not needed
    # Import numpy here to avoid circular imports if this script
    # is imported as a module elsewhere
    import numpy as np

    main()
