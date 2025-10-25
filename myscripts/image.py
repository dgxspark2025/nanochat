"""
Image classification dataset download and categorization script.
"""

from pathlib import Path
from typing import Optional
import torch
from torchvision import datasets, transforms


DEFAULT_DATA_DIR = Path.home() / ".cache" / "nanochat" / "image_data"


def download_cifar10(data_dir: Optional[Path] = None) -> Path:
    """
    Download CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store the dataset. Defaults to ~/.cache/nanochat/image_data
        
    Returns:
        Path to the dataset directory
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading CIFAR-10 to {data_dir}")
    
    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=None
    )
    
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=None
    )
    
    print(f"Dataset downloaded successfully")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    return data_dir


def categorize_images(data_dir: Optional[Path] = None, split: str = "train"):
    """
    Load and categorize CIFAR-10 images by class.
    
    Args:
        data_dir: Directory containing the dataset
        split: Dataset split ("train" or "test")
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    is_train = split == "train"
    dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=is_train,
        download=False,
        transform=transform
    )
    
    class_counts = {}
    print(f"\nCategorizing {len(dataset)} images...")
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nImage categorization summary:")
    print("-" * 40)
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name:15s}: {count:5d} images")
    print("-" * 40)
    print(f"Total: {sum(class_counts.values())} images")


def main():
    """Main entrypoint for image dataset operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and categorize image datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Directory to store datasets (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load dataset without categorization"
    )
    
    args = parser.parse_args()
    
    download_cifar10(args.data_dir)
    
    if not args.load_only:
        categorize_images(args.data_dir, args.split)
    

if __name__ == "__main__":
    main()
