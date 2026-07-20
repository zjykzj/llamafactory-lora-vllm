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
    model="model",
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

# CIFAR100 测试
python eval/eval_cifar100.py --max-samples 200

# 自定义 vLLM 地址
python eval/eval_cifar10.py --base-url http://127.0.0.1:8080/v1
```
