#!/usr/bin/env python3
"""
Download CIFAR10/CIFAR100 and save images as PNG files.

Usage:
    python data/download_cifar.py --dataset cifar10
    python data/download_cifar.py --dataset cifar100 --subset 500
    python data/download_cifar.py --dataset cifar10 --local-path /path/to/local/dataset
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100

# Project root
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
IMAGES_DIR = ROOT / "data" / "images"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]


def save_image(array: np.ndarray, path: Path) -> None:
    """Save a CIFAR image (32x32 RGB or PIL Image) as PNG."""
    if isinstance(array, Image.Image):
        array.save(str(path))
    else:
        img = Image.fromarray(array)
        img.save(str(path))


def copy_local_archive(local_dir: str, archive_name: str, dest_dir: Path) -> None:
    """Copy a dataset archive from a local directory to RAW_DIR.

    If the archive already exists at the destination, skip copying.
    If not found in local_dir, print a warning and let torchvision download it.
    """
    src = Path(local_dir) / archive_name
    if not src.exists():
        print(f"  Warning: {archive_name} not found in {local_dir}, will attempt download")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / archive_name
    if dest.exists():
        print(f"  {archive_name} already exists in {dest_dir}, skip copy")
        return
    print(f"  Copying {src} -> {dest}")
    shutil.copy2(src, dest)


def download_cifar10(subset: int | None = None, local_dir: str | None = None) -> None:
    """Download CIFAR10 and save images to data/images/cifar10/."""
    classes = CIFAR10_CLASSES

    if local_dir:
        copy_local_archive(local_dir, "cifar-10-python.tar.gz", RAW_DIR)

    for split, train in [("train", True), ("test", False)]:
        dataset = CIFAR10(root=str(RAW_DIR), train=train, download=True)
        out_dir = IMAGES_DIR / "cifar10" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Group indices by class
        class_indices = {c: [] for c in range(10)}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        count = 0
        for label, indices in class_indices.items():
            selected = indices[:subset] if subset else indices
            for i, idx in enumerate(selected):
                img, _ = dataset[idx]
                fname = f"{label}_{classes[label]}_{i:05d}.png"
                save_image(img, out_dir / fname)
                count += 1

        print(f"  CIFAR10 {split}: saved {count} images to {out_dir}")

    # Save class mapping
    mapping = {i: name for i, name in enumerate(classes)}
    mapping_path = IMAGES_DIR / "cifar10" / "class_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)


def download_cifar100(subset: int | None = None, local_dir: str | None = None) -> None:
    """Download CIFAR100 and save images to data/images/cifar100/."""
    classes = CIFAR100_CLASSES

    if local_dir:
        copy_local_archive(local_dir, "cifar-100-python.tar.gz", RAW_DIR)

    for split, train in [("train", True), ("test", False)]:
        dataset = CIFAR100(root=str(RAW_DIR), train=train, download=True)
        out_dir = IMAGES_DIR / "cifar100" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Group indices by class
        class_indices = {c: [] for c in range(100)}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        count = 0
        for label, indices in class_indices.items():
            selected = indices[:subset] if subset else indices
            for i, idx in enumerate(selected):
                img, _ = dataset[idx]
                fname = f"{label}_{classes[label]}_{i:05d}.png"
                save_image(img, out_dir / fname)
                count += 1

        print(f"  CIFAR100 {split}: saved {count} images to {out_dir}")

    # Save class mapping
    mapping = {i: name for i, name in enumerate(classes)}
    mapping_path = IMAGES_DIR / "cifar100" / "class_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR datasets and save as PNG")
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "all"],
        default="cifar10", help="Which dataset to download"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Number of images per class (default: all). "
             "Recommended: 200-500 for quick fine-tuning experiments."
    )
    parser.add_argument(
        "--local-path", type=str, default=None,
        help="Path to a local directory containing pre-downloaded CIFAR tar.gz files "
             "(e.g. cifar-10-python.tar.gz, cifar-100-python.tar.gz). "
             "If provided, copies them to RAW_DIR instead of downloading."
    )
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    if args.dataset in ("cifar10", "all"):
        print(f"Downloading CIFAR10 (subset={args.subset})...")
        download_cifar10(subset=args.subset, local_dir=args.local_path)

    if args.dataset in ("cifar100", "all"):
        print(f"Downloading CIFAR100 (subset={args.subset})...")
        download_cifar100(subset=args.subset, local_dir=args.local_path)


if __name__ == "__main__":
    main()
