# vLLM 部署

## 概述

[vLLM](https://github.com/vllm-project/vllm) 是一个高性能 LLM 推理引擎，核心特性：

- **PagedAttention**：高效管理 KV cache，提升吞吐量
- **Continuous batching**：动态合并请求，降低延迟
- **OpenAI 兼容 API**：可直接替换 OpenAI 客户端
- **多模态支持**：图片输入等
- **原生 LoRA 支持**：`--enable-lora --lora-modules` 直接加载，无需合并

## 部署流程

### 方式一：LoRA 适配器直接部署（推荐）

vLLM 原生支持 LoRA，无需合并模型：

```bash
vllm serve /path/to/Qwen3.5-2B \
    --enable-lora \
    --lora-modules cifar10=models/lora/cifar10 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

多个 adapter 可同时加载，通过请求中的 `model` 字段动态切换：

```bash
vllm serve /path/to/Qwen3.5-2B \
    --enable-lora \
    --lora-modules cifar10=models/lora/cifar10,cifar100=models/lora/cifar100 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

请求时指定 model 名称：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="cifar10",  # 对应 --lora-modules 中定义的名称
    messages=[{"role": "user", "content": "Hello"}],
)
```

或使用本项目的脚本：

```bash
bash serve/serve.sh --lora models/lora/cifar10
```

### 方式二：合并后部署

如果需要部署合并后的完整模型（单任务、无需动态切换 adapter）：

```bash
# 步骤 1：合并 LoRA 到基座模型
llamafactory-cli export configs/cifar10_merge.yaml

# 步骤 2：启动 vLLM
vllm serve models/merged/cifar10 \
    --served-model-name cifar10 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

或使用本项目的脚本（自动取目录 basename 作为模型名）：

```bash
bash serve/serve.sh --model models/merged/cifar10
# 启动后显示: Use --model cifar10 when running eval scripts.
```

### 验证

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

## 关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--host` | 绑定地址 | `0.0.0.0` |
| `--port` | 端口 | `8000` |
| `--trust-remote-code` | 允许自定义模型代码 | 必需（Qwen） |
| `--api-key` | API 认证密钥 | `not-needed`（本地） |
| `--gpu-memory-utilization` | GPU 显存使用比例 | `0.9`（默认） |
| `--max-model-len` | 最大序列长度 | 根据模型设置 |
| `--tensor-parallel-size` | 多卡并行 | `1`（单卡） |
| `--served-model-name` | 对外暴露的模型名 | 目录 basename（如 `cifar10`） |
| `--enable-lora` | 启用 LoRA 支持 | LoRA 模式必需 |
| `--lora-modules` | LoRA 适配器映射 | `name=path` 格式 |

## 场景调参指南

### 低显存部署大模型（如 Qwen3.5-9B）

部署 9B 或更大模型时显存不足，按以下顺序逐步调整参数（每个步骤独立生效，可组合）：

**步骤 1：降低 GPU 显存利用率**

```bash
vllm serve ./models/merged/cifar10_qwen3.5-9B \
    --served-model-name cifar10_qwen3.5-9B \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --gpu-memory-utilization 0.7
```

`--gpu-memory-utilization` 控制 vLLM 最多使用多少比例的显存。从默认 0.9 降到 0.6~0.7 释放了 KV cache 之外的余地，但可用 KV cache 变小，可能影响并发。

**步骤 2：限制最大序列长度（推荐，效果最显著）**

```bash
vllm serve ./models/merged/cifar10_qwen3.5-9B \
    --served-model-name cifar10_qwen3.5-9B \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048
```

`--max-model-len` 直接限制 KV cache 预分配的最大长度。**CIFAR 分类任务的 prompt 很短**（一张图 + 简单指令，通常 < 512 tokens），设为 1024~2048 完全够用，可节省数 GB 显存。

**步骤 3：限制并发序列数**

```bash
vllm serve ./models/merged/cifar10_qwen3.5-9B \
    --served-model-name cifar10_qwen3.5-9B \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048 \
    --max-num-seqs 8
```

`--max-num-seqs` 限制同时处理的请求数，减少 KV block 分配。eval 脚本通常是单条请求逐个发送，设为 8~16 对延迟影响很小。

**步骤 4：禁用 CUDA graph（最后手段）**

```bash
vllm serve ./models/merged/cifar10_qwen3.5-9B \
    --served-model-name cifar10_qwen3.5-9B \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --enforce-eager
```

`--enforce-eager` 禁用 CUDA graph 加速，牺牲约 5% 推理速度，释放数百 MB 显存。仅在前面步骤不够时使用。

**推荐组合（对 CIFAR 分类任务）**：

```bash
# 使用 serve.sh
bash serve/serve.sh \
    --model ./models/merged/cifar10_qwen3.5-9B \
    --gpu-memory 0.7 \
    --max-model-len 2048
```

| 参数 | 作用 | 9B 建议值 |
|------|------|----------|
| `--gpu-memory-utilization` | GPU 显存使用比例 | `0.7` |
| `--max-model-len` | 最大序列长度 | `2048`（CIFAR 场景够用） |
| `--max-num-seqs` | 最大并发序列数 | `8`（eval 单请求场景不敏感） |
| `--enforce-eager` | 禁用 CUDA graph | 最后手段，释放数百 MB |

### 多卡并行

有多张 GPU 时，使用 `--tensor-parallel-size` 将模型拆分到多张卡上：

```bash
# 2 卡并行
bash serve/serve.sh \
    --model ./models/merged/cifar10_qwen3.5-9B \
    --tensor-parallel 2 \
    --gpu-memory 0.9
```

每张卡的显存占用约为单卡的 `1 / N`。注意多卡通信有开销，实际吞吐不一定线性增长。

### 高并发 / 高吞吐

需要高吞吐时，调大并发相关参数（前提是显存足够）：

```bash
vllm serve ./models/merged/cifar10 \
    --served-model-name cifar10 \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256
```

| 参数 | 低负载默认 | 高并发建议值 |
|------|-----------|-------------|
| `--gpu-memory-utilization` | `0.9` | `0.95` |
| `--max-num-seqs` | `256` | `256`~`512` |
| `--max-model-len` | 模型最大值 | 按实际需要（CIFAR 设 `2048` 即可） |

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 文本请求
response = client.chat.completions.create(
    model="cifar10",  # 合并模式用 served-model-name，LoRA 模式用 adapter 名称
    messages=[{"role": "user", "content": "Hello"}],
)
```

## 参考

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora.html)
- [vLLM OpenAI Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [serve/README.md](../serve/README.md) — 本项目部署脚本详细文档
