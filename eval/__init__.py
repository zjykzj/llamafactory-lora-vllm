# -*- coding: utf-8 -*-
"""
Shared utilities for CIFAR evaluation via vLLM OpenAI-compatible API.
"""
import base64
import io
import re
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image


def normalize_label(s: str) -> str:
    """Normalize a class label for comparison.

    Collapses underscores, hyphens, and whitespace into a single space,
    then lowercases and strips. This ensures ``"pickup_truck"``,
    ``"pickup truck"``, and ``"Pickup Truck"`` all match.
    """
    return re.sub(r"[\s_\-]+", " ", s.strip().lower()).strip()


def image_to_base64(image: Image.Image | np.ndarray) -> str:
    """Convert PIL Image or numpy array to base64 data URL string."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def classify_image(
    client: OpenAI,
    model: str,
    image: Image.Image | np.ndarray,
    class_list: list[str] | None = None,
) -> str:
    """Send an image to vLLM and return the predicted class name.

    If class_list is provided, it is included in the prompt to constrain
    the output space (e.g. "Classify this image into one of these
    categories: ..."). Both CIFAR10 and CIFAR100 pass their class lists.
    """
    if class_list:
        class_str = ", ".join(class_list)
        text = (
            f"Classify this image into one of these categories: {class_str}. "
            "Answer with only the class name."
        )
    else:
        text = "Classify this image. Answer with only the class name."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_base64(image)},
                    },
                    {"type": "text", "text": text},
                ],
            }
        ],
        max_tokens=32,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return normalize_label(response.choices[0].message.content)


def compute_accuracy(
    predictions: list[str],
    ground_truth: list[str],
) -> dict[str, Any]:
    """Compute accuracy metrics. Both predictions and ground truth are
    normalized before comparison (see :func:`normalize_label`)."""
    correct = sum(
        1 for p, g in zip(predictions, ground_truth)
        if normalize_label(p) == normalize_label(g)
    )
    total = len(ground_truth)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def evaluate_dataset(
    client: OpenAI,
    model: str,
    samples: list[tuple[Image.Image, int]],
    class_names: list[str],
    class_list: list[str] | None = None,
    workers: int = 1,
) -> tuple[list[str], list[str]]:
    """Evaluate a dataset with optional multi-worker concurrency.

    Args:
        client: OpenAI-compatible client.
        model: Model name as served by vLLM.
        samples: List of (image, label_id) tuples.
        class_names: List mapping label_id → class name string.
        class_list: If provided, included in the prompt to constrain outputs
                    (e.g. CIFAR10). Omit for datasets with many classes.
        workers: Number of concurrent threads (1 = sequential, the default).

    Returns:
        (predictions, ground_truth) as lists of class name strings, aligned.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    results: dict[int, str] = {}
    ground_truth = [class_names[label] for _, label in samples]

    def _classify(idx: int, image: Image.Image) -> tuple[int, str]:
        try:
            return idx, classify_image(client, model, image, class_list)
        except Exception:
            return idx, "<error>"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_classify, i, img): i
            for i, (img, _) in enumerate(samples)
        }
        with tqdm(total=len(futures), desc="Evaluating") as pbar:
            for future in as_completed(futures):
                idx, pred = future.result()
                results[idx] = pred
                pbar.update(1)

    # Reconstruct predictions in original order
    predictions = [results[i] for i in range(len(samples))]
    return predictions, ground_truth
