#!/usr/bin/env python3
"""
Evaluate fine-tuned model on CIFAR100 via vLLM OpenAI-compatible API.

Usage:
    python eval/eval_cifar100.py
    python eval/eval_cifar100.py --base-url http://localhost:8000/v1 --max-samples 200
"""
import argparse
from pathlib import Path

from openai import OpenAI
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from eval import classify_image, compute_accuracy

ROOT = Path(__file__).resolve().parent.parent

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
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR100 classification via vLLM API"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL"
    )
    parser.add_argument(
        "--model", default="cifar100",
        help="Model name as served by vLLM. For merged mode this is the "
             "directory basename (e.g. cifar100, cifar100_qwen3.5-0.8B). "
             "For LoRA mode this is the adapter name shown at startup."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max test samples to evaluate (default: all 10000)"
    )
    parser.add_argument(
        "--api-key", default="not-needed",
        help="API key (vLLM default is 'not-needed')"
    )
    args = parser.parse_args()

    # Load CIFAR100 test set
    data_dir = str(ROOT / "data" / "raw")
    dataset = CIFAR100(root=data_dir, train=False, download=True)
    print(f"Loaded CIFAR100 test set: {len(dataset)} images")

    samples = list(dataset) if args.max_samples is None else list(dataset)[:args.max_samples]
    print(f"Evaluating {len(samples)} samples...")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    predictions = []
    ground_truth = []

    for img, label in tqdm(samples):
        true_class = CIFAR100_CLASSES[label]
        try:
            # CIFAR100 has 100 classes — omit class_list to avoid prompt bloat.
            # The model learns class names from training, no list needed at inference.
            pred = classify_image(client, args.model, img)
        except Exception as e:
            print(f"\nError on sample {label}: {e}")
            pred = "<error>"
        predictions.append(pred)
        ground_truth.append(true_class)

    metrics = compute_accuracy(predictions, ground_truth)
    print(f"\nCIFAR100 Results: {metrics['correct']}/{metrics['total']} correct")
    print(f"Accuracy: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()
