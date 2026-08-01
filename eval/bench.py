#!/usr/bin/env python3
"""
Benchmark vLLM / LLaMA Factory API latency and throughput.

Measures per-request latency, throughput, and token usage by sending
images to an OpenAI-compatible API endpoint.

Usage:
    python eval/bench.py
    python eval/bench.py --num-samples 100 --model cifar10
    python eval/bench.py --dataset cifar100 --base-url http://localhost:8080/v1
"""
import argparse
import statistics
import time
from pathlib import Path

from openai import OpenAI
from torchvision.datasets import CIFAR10, CIFAR100

from eval import image_to_base64, normalize_label

ROOT = Path(__file__).resolve().parent.parent

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
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def _percentile(values: list[float], pct: float) -> float:
    """Return the pct-th percentile of sorted values (0–100)."""
    if not values:
        return 0.0
    k = (len(values) - 1) * pct / 100
    f = int(k)
    c = k - f
    if f + 1 < len(values):
        return values[f] + c * (values[f + 1] - values[f])
    return values[f]


def _fmt(seconds: float) -> str:
    """Human-readable latency."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    else:
        return f"{seconds:.2f} s"


def _call_api(
    client: OpenAI,
    model: str,
    image,
    class_list: list[str] | None = None,
) -> tuple[str, float, dict | None]:
    """Send one classification request. Returns (predicted_class, elapsed_seconds, usage_dict_or_None).

    If class_list is provided (e.g. CIFAR10), it is included in the prompt.
    For datasets with many classes (e.g. CIFAR100), omit to avoid prompt bloat.
    """
    if class_list:
        class_str = ", ".join(class_list)
        text = (
            f"Classify this image into one of these categories: {class_str}. "
            "Answer with only the class name."
        )
    else:
        text = "Classify this image. Answer with only the class name."

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_base64(image)}},
                {"type": "text", "text": text},
            ],
        }],
        max_tokens=32,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed = time.perf_counter() - t0
    pred = normalize_label(response.choices[0].message.content)
    usage = None
    if hasattr(response, "usage") and response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return pred, elapsed, usage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM/LlaMA Factory API latency and throughput"
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="cifar10")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--api-key", default="not-needed")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────

    DatasetCls = CIFAR10 if args.dataset == "cifar10" else CIFAR100
    class_list = CIFAR10_CLASSES if args.dataset == "cifar10" else CIFAR100_CLASSES

    data_dir = str(ROOT / "data" / "raw")
    dataset = DatasetCls(root=data_dir, train=False, download=True)
    images = list(dataset)

    total_needed = args.warmup + args.num_samples
    if total_needed > len(images):
        total_needed = len(images)
        args.num_samples = total_needed - args.warmup

    warmup_imgs = images[:args.warmup]
    bench_imgs = images[args.warmup:total_needed]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=120.0)

    # ── Warmup ─────────────────────────────────────────────────

    if args.warmup > 0:
        print(f"Warming up ({args.warmup} requests)...", end=" ", flush=True)
        warmup_errors = 0
        for img, _ in warmup_imgs:
            try:
                _call_api(client, args.model, img, class_list)
            except Exception as e:
                warmup_errors += 1
                if warmup_errors <= 1:
                    print(f"\n  [warmup] error: {e}", end="", flush=True)
        if warmup_errors == args.warmup:
            print("\n  [warmup] ALL requests failed — check that the server is running.\n")
            return
        print("done.\n")

    # ── Benchmark ──────────────────────────────────────────────

    print(f"Benchmark: {args.dataset.upper()}  |  {args.base_url}  |  model={args.model}")
    print(f"Running {args.num_samples} requests...\n")

    latencies: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    errors = 0
    total_start = time.perf_counter()

    for i, (img, _) in enumerate(bench_imgs):
        try:
            pred, elapsed, usage = _call_api(client, args.model, img)
            latencies.append(elapsed)
            if usage:
                prompt_tokens.append(usage["prompt_tokens"])
                completion_tokens.append(usage["completion_tokens"])
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [{i + 1}] error: {e}")

        if (i + 1) % max(1, args.num_samples // 5) == 0:
            print(f"  [{i + 1}/{args.num_samples}]  "
                  f"avg={_fmt(sum(latencies) / len(latencies)) if latencies else '—'}  "
                  f"last={_fmt(latencies[-1]) if latencies else '—'}")

    total_time = time.perf_counter() - total_start

    if not latencies:
        print("\nNo successful requests.")
        return

    latencies.sort()

    # ── Report ─────────────────────────────────────────────────

    print()
    print("=" * 52)
    print("  Benchmark Results")
    print("=" * 52)
    print(f"  Backend:     {args.base_url}")
    print(f"  Model:       {args.model}")
    print(f"  Succeeded:   {len(latencies)}")
    if errors:
        print(f"  Errors:      {errors}")
    print()
    print("  ── Latency ──")
    print(f"  min          {_fmt(latencies[0]):>8}")
    print(f"  avg          {_fmt(sum(latencies) / len(latencies)):>8}")
    print(f"  p50          {_fmt(_percentile(latencies, 50)):>8}")
    print(f"  p95          {_fmt(_percentile(latencies, 95)):>8}")
    print(f"  p99          {_fmt(_percentile(latencies, 99)):>8}")
    print(f"  max          {_fmt(latencies[-1]):>8}")
    print()
    print("  ── Throughput ──")
    print(f"  total        {_fmt(total_time):>8}")
    req_per_sec = len(latencies) / total_time
    print(f"  throughput   {req_per_sec:>7.1f} req/s (sequential)")
    print()
    if prompt_tokens:
        print("  ── Tokens (per request avg) ──")
        print(f"  prompt       {sum(prompt_tokens) / len(prompt_tokens):>7.0f}")
        print(f"  completion   {sum(completion_tokens) / len(completion_tokens):>7.0f}")
        total_tok = sum(prompt_tokens) + sum(completion_tokens)
        print(f"  total used   {total_tok:>7d}")
    print("=" * 52)


if __name__ == "__main__":
    main()
