# Made by: Jonathan Mikler on 2025-02-10
import datasets
import torch
from torchvision import transforms
from typing import Dict, Tuple, Union

from utils.logging_utils import get_logger_ready, yellow_txt

logger = get_logger_ready("dataset_utils")


def get_cifar100_dataset() -> datasets.DatasetDict:
    """
    Load CIFAR-100 dataset from Huggingface datasets.
    Returns train and test splits.
    """
    logger.info(yellow_txt("Loading CIFAR-100 dataset..."))

    # Load dataset
    try:
        dataset = datasets.load_dataset(
            path="cifar100",
            token=None,
            # task=task_arg,
            cache_dir="./tmp/data/cifar100",
        )
    except Exception as e:
        logger.error(f"Failed to load CIFAR-100 dataset: {e}")
        raise

    logger.debug(f"Dataset splits available: {dataset.keys()}")
    logger.debug(f"Training examples: {len(dataset['train'])}")
    logger.debug(f"Test examples: {len(dataset['test'])}")

    return dataset


def get_transforms(split: str = "train", size: int = 224) -> transforms.Compose:
    """
    Get transforms for the dataset based on split type.
    Args:
        split: Either 'train' or 'test'
        size: Size to resize images to
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # taken from lgvit image_processor
        std=[0.229, 0.224, 0.225],  # taken from lgvit image_processor
    )

    transform_dict = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }

    transform = transform_dict[split]

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
    num_examples: int = None,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Prepare dataset by adding transforms and label names
    """
    logger.info(yellow_txt("Preparing dataset..."))
    train_transforms = get_transforms(split="train")
    test_transforms = get_transforms(split="test")

    def apply_transforms(
        examples: Dict,
        transforms,
        source_dataset: Union[datasets.Dataset, torch.utils.data.Subset],
    ) -> Dict:
        """
        Apply transforms across a batch.

        Args:
            examples: Dictionary of examples to transform
            transforms: Transform functions to apply
            source_dataset: The source dataset to get label names from
        """
        examples["pixel_values"] = [
            transforms(img.convert("RGB")) for img in examples["img"]
        ]
        # Add label names using the new get_label_name function
        examples["fine_label_name"] = [
            get_label_name(source_dataset, label) for label in examples["fine_label"]
        ]
        return examples

    # Set the transforms
    logger.info("Applying transforms to training set...")
    dataset["train"].set_transform(
        lambda x: apply_transforms(x, train_transforms, dataset["train"])
    )

    logger.info("Applying transforms to test set...")
    dataset["test"].set_transform(
        lambda x: apply_transforms(x, test_transforms, dataset["test"])
    )

    if num_examples is not None:
        logger.info(f"Using {num_examples} examples from each dataset split")
        get_random = True
        for split in ["train", "test"]:
            if get_random:
                indices = torch.randperm(len(dataset[split]))[:num_examples]
                logger.warning(
                    f"Retrieving random {num_examples} examples for the {split} split"
                )
            else:
                indices = torch.arange(num_examples)
            dataset[split] = torch.utils.data.Subset(dataset[split], indices)

    return dataset["train"], dataset["test"]


def get_label_name(
    dataset: Union[torch.utils.data.Subset, datasets.DatasetDict], label_idx: int
) -> str:
    """
    Get label name from dataset handling both full dataset and subset cases.

    Args:
        dataset: Either a HuggingFace Dataset, PyTorch Dataset, or a Subset wrapping either
        label_idx: Integer index of the label to look up

    Returns:
        str: The human-readable label name

    Raises:
        AttributeError: If the dataset structure doesn't contain feature names
        ValueError: If the label_idx is invalid
    """
    try:
        if hasattr(dataset, "features"):
            name = dataset.features["fine_label"].names[label_idx]
        elif hasattr(dataset, "dataset"):  # For Subset objects
            name = dataset.dataset.features["fine_label"].names[label_idx]
        else:
            raise AttributeError(
                "Dataset structure doesn't contain feature names. "
                "Expected either dataset.features or dataset.dataset.features"
            )
    except IndexError as e:
        raise ValueError(f"Invalid label index {label_idx}") from e

    return name
