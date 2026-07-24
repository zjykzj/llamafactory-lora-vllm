#!/usr/bin/env python3
"""
Evaluate fine-tuned model on CIFAR10 via vLLM OpenAI-compatible API.

Usage:
    python eval/eval_cifar10.py
    python eval/eval_cifar10.py --base-url http://localhost:8000/v1 --max-samples 200
"""
import argparse
from pathlib import Path

from openai import OpenAI
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from eval import classify_image, compute_accuracy

ROOT = Path(__file__).resolve().parent.parent

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR10 classification via vLLM API"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL"
    )
    parser.add_argument(
        "--model", default="cifar10",
        help="Model name as served by vLLM. For merged mode this is the "
             "directory basename (e.g. cifar10, cifar10_qwen3.5-0.8b). "
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

    # Load CIFAR10 test set
    data_dir = str(ROOT / "data" / "raw")
    dataset = CIFAR10(root=data_dir, train=False, download=True)
    print(f"Loaded CIFAR10 test set: {len(dataset)} images")

    samples = list(dataset) if args.max_samples is None else list(dataset)[:args.max_samples]
    print(f"Evaluating {len(samples)} samples...")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    predictions = []
    ground_truth = []

    for img, label in tqdm(samples):
        true_class = CIFAR10_CLASSES[label]
        try:
            pred = classify_image(client, args.model, img, CIFAR10_CLASSES)
        except Exception as e:
            print(f"\nError on sample {label}: {e}")
            pred = "<error>"
        predictions.append(pred)
        ground_truth.append(true_class)

    metrics = compute_accuracy(predictions, ground_truth)
    print(f"\nCIFAR10 Results: {metrics['correct']}/{metrics['total']} correct")
    print(f"Accuracy: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()
