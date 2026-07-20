#!/usr/bin/env python3
"""
Build ShareGPT-format instruction data from CIFAR PNG images.

Output format (LLamaFactory multimodal):
    [
      {
        "messages": [
          {"role": "user", "content": "<image>Classify this image into..."},
          {"role": "assistant", "content": "airplane"}
        ],
        "images": ["data/images/cifar10/train/0_airplane_00001.png"]
      },
      ...
    ]

Usage:
    python data/build_instructions.py --dataset cifar10
    python data/build_instructions.py --dataset cifar100 --max-train 5000
"""
import argparse
import json
import os
import random
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
PROCESSED_DIR = ROOT / "data" / "processed"

# Prompt templates for variety
PROMPT_TEMPLATES = {
    "cifar10": {
        "class_list": "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck",
        "prompts": [
            "<image>Classify this image into one of these categories: {class_list}. Answer with only the class name.",
            "<image>What object is shown in this image? Choose from: {class_list}.",
            "<image>Identify the main object in this picture. Options: {class_list}. Respond with just the category name.",
            "<image>This is a CIFAR-10 image. Which of the following classes does it belong to? {class_list}",
        ],
    },
    "cifar100": {
        "class_list": (
            "apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, "
            "bicycle, bottle, bowl, boy, bridge, bus, butterfly, camel, "
            "can, castle, caterpillar, cattle, chair, chimpanzee, clock, "
            "cloud, cockroach, couch, crab, crocodile, cup, dinosaur, "
            "dolphin, elephant, flatfish, forest, fox, girl, hamster, "
            "house, kangaroo, keyboard, lamp, lawn_mower, leopard, lion, "
            "lizard, lobster, man, maple_tree, motorcycle, mountain, mouse, "
            "mushroom, oak_tree, orange, orchid, otter, palm_tree, pear, "
            "pickup_truck, pine_tree, plain, plate, poppy, porcupine, "
            "possum, rabbit, raccoon, ray, road, rocket, rose, "
            "sea, seal, shark, shrew, skunk, skyscraper, snail, snake, "
            "spider, squirrel, streetcar, sunflower, sweet_pepper, table, "
            "tank, telephone, television, tiger, tractor, train, trout, "
            "tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm"
        ),
        "prompts": [
            "<image>Classify this image into one of these categories: {class_list}. Answer with only the class name.",
            "<image>What object is shown in this image? Choose from the CIFAR-100 categories. Respond with just the category name.",
            "<image>Identify the main object in this picture from the CIFAR-100 dataset. Options include: {class_list}. Answer with only the class name.",
        ],
    },
}


def build_dataset(dataset_name: str, max_train: int | None = None) -> None:
    """Build ShareGPT JSON files for a CIFAR dataset."""
    templates = PROMPT_TEMPLATES[dataset_name]
    class_list = templates["class_list"]
    prompts = templates["prompts"]

    images_dir = IMAGES_DIR / dataset_name

    # Load class mapping
    with open(images_dir / "class_mapping.json") as f:
        class_mapping = json.load(f)  # {0: "airplane", 1: "automobile", ...}

    for split in ["train", "test"]:
        split_dir = images_dir / split
        if not split_dir.exists():
            print(f"  Skip {split}: directory not found ({split_dir})")
            continue

        # Collect image paths and labels
        samples = []
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".png"):
                continue
            # Filename format: {label_id}_{class_name}_{index}.png
            parts = fname.split("_", 1)
            label_id = int(parts[0])
            samples.append((str(split_dir / fname), label_id, class_mapping[str(label_id)]))

        random.shuffle(samples)

        if max_train and split == "train" and len(samples) > max_train:
            samples = samples[:max_train]

        # Build ShareGPT records
        records = []
        for img_path, label_id, class_name in samples:
            prompt = random.choice(prompts).format(class_list=class_list)
            records.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": class_name},
                ],
                "images": [img_path],
            })

        # Write JSON
        out_path = PROCESSED_DIR / f"{dataset_name}_{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"  {dataset_name} {split}: {len(records)} samples -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ShareGPT instruction data from CIFAR images"
    )
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "all"],
        default="cifar10", help="Which dataset to build"
    )
    parser.add_argument(
        "--max-train", type=int, default=None,
        help="Max training samples (default: all). CIFAR10 has 50000 train images."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if args.dataset in ("cifar10", "all"):
        print(f"Building CIFAR10 instructions (max_train={args.max_train})...")
        build_dataset("cifar10", max_train=args.max_train)

    if args.dataset in ("cifar100", "all"):
        print(f"Building CIFAR100 instructions (max_train={args.max_train})...")
        build_dataset("cifar100", max_train=args.max_train)

    print("Done.")


if __name__ == "__main__":
    main()
