# Eval

调用 vLLM OpenAI 兼容接口评估分类模型准确率。

## 原理

Qwen3.5-2B 是多模态模型，支持图像+文本输入。评估时：

1. 加载 CIFAR 测试集图片
2. 将图片编码为 base64，通过 `image_url` 传入 API
3. 模型返回分类结果文字
4. 与真实标签比对，计算准确率

**直接输入图片，无需额外预处理。**

## API 请求示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="cifar10",  # 对应 serve 启动时显示的 Model name
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "Classify this image into..."},
        ]
    }],
    max_tokens=32,
    temperature=0.0,
)
```

## 运行

```bash
# 完整测试（CIFAR10 10000 张）
python eval/eval_cifar10.py

# 仅测试 100 张（快速验证）
python eval/eval_cifar10.py --max-samples 100

# 指定模型名（与 serve 启动时显示的 Model name 一致）
python eval/eval_cifar10.py --model cifar10_qwen3.5-0.8b --max-samples 100

# CIFAR100 测试
python eval/eval_cifar100.py --max-samples 200

# 自定义 vLLM 地址
python eval/eval_cifar10.py --base-url http://127.0.0.1:8080/v1
```

## Benchmark（延迟 / 吞吐）

`bench.py` 测量 API 的延迟和吞吐，与准确率评估独立。

```bash
# 默认 50 个样本 + 3 次 warmup
python eval/bench.py

# 指定模型和样本数
python eval/bench.py --model cifar10_qwen3.5-0.8b --num-samples 100

# 跨后端对比
python eval/bench.py --base-url http://localhost:8000/v1   # vLLM
python eval/bench.py --base-url http://localhost:8080/v1   # LLaMA Factory API

# CIFAR100
python eval/bench.py --dataset cifar100
```

示例输出：

```
Benchmark: CIFAR10  |  http://localhost:8000/v1  |  model=cifar10
Running 50 requests...

  [10/50]  avg=820 ms  last=795 ms
  [20/50]  avg=815 ms  last=830 ms
  ...

====================================================
  Benchmark Results
====================================================
  Backend:     http://localhost:8000/v1
  Model:       cifar10
  Succeeded:   50

  ── Latency ──
  min            752 ms
  avg            818 ms
  p50            810 ms
  p95            920 ms
  p99            980 ms
  max           1050 ms

  ── Throughput ──
  total         41.2 s
  throughput     1.2 req/s (sequential)

  ── Tokens (per request avg) ──
  prompt            1180
  completion           2
  total used       59100
====================================================
```
