# Made by: Jonathan Mikler on 2025-02-10
import datasets
import torch
from torchvision import transforms
from typing import Dict, Tuple

from utils.logging_utils import get_logger_ready, yellow_txt

logger = get_logger_ready("dataset_utils")


def get_cifar100_dataset() -> datasets.DatasetDict:
    """
    Load CIFAR-100 dataset from Huggingface datasets.
    Returns train and test splits.
    """
    logger.info(yellow_txt("Loading CIFAR-100 dataset..."))

    # Load dataset
    dataset = datasets.load_dataset(
        path="cifar100",
        token=None,
        # task=task_arg,
        cache_dir="./tmp/data/cifar100",
    )

    logger.info(f"Dataset splits available: {dataset.keys()}")
    logger.info(f"Training examples: {len(dataset['train'])}")
    logger.info(f"Test examples: {len(dataset['test'])}")

    return dataset


def get_transforms(split: str = "train", size: int = 224) -> transforms.Compose:
    """
    Get transforms for the dataset based on split type.
    Args:
        split: Either 'train' or 'test'
        size: Size to resize images to
    """
    normalize = transforms.Normalize(
        # TODO: Check these values
        mean=[0.485, 0.456, 0.406],  # taken from lgvit image_processor
        std=[0.229, 0.224, 0.225],  # taken from lgvit image_processor
    )

    if split == "train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transform


def collate_fn(examples):
    """
    Custom collate function for DataLoader.
    Properly handles converting PIL images to tensors and stacking them.
    Also includes label names alongside numeric labels.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["fine_label"] for example in examples])
    label_names = [example["fine_label_name"] for example in examples]
    return {"pixel_values": pixel_values, "labels": labels, "label_names": label_names}


def prepare_dataset(
    dataset: datasets.DatasetDict,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Prepare dataset by adding transforms and label names
    """
    logger.info(yellow_txt("Preparing dataset..."))
    train_transforms = get_transforms(split="train")
    test_transforms = get_transforms(split="test")

    def apply_transforms(examples: Dict, transforms) -> Dict:
        """Apply transforms across a batch."""
        examples["pixel_values"] = [
            transforms(img.convert("RGB")) for img in examples["img"]
        ]
        # Add label names to the examples
        examples["fine_label_name"] = [
            dataset["train"].features["fine_label"].names[label]
            for label in examples["fine_label"]
        ]
        return examples

    # Set the transforms
    logger.info("Applying transforms to training set...")
    dataset["train"].set_transform(lambda x: apply_transforms(x, train_transforms))

    logger.info("Applying transforms to test set...")
    dataset["test"].set_transform(lambda x: apply_transforms(x, test_transforms))

    return dataset["train"], dataset["test"]
