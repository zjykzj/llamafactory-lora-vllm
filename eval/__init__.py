# -*- coding: utf-8 -*-
"""
Shared utilities for CIFAR evaluation via vLLM OpenAI-compatible API.
"""
import base64
import io
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image


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
    class_list: list[str],
) -> str:
    """Send an image to vLLM and return the predicted class name."""
    class_str = ", ".join(class_list)
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
                    {
                        "type": "text",
                        "text": (
                            f"Classify this image into one of these categories: {class_str}. "
                            "Answer with only the class name, nothing else."
                        ),
                    },
                ],
            }
        ],
        max_tokens=32,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip().lower()


def compute_accuracy(
    predictions: list[str],
    ground_truth: list[str],
) -> dict[str, Any]:
    """Compute accuracy metrics."""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total = len(ground_truth)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
    }
