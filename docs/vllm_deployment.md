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
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

或使用本项目的脚本：

```bash
bash serve/serve.sh models/merged/cifar10
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
| `--enable-lora` | 启用 LoRA 支持 | LoRA 模式必需 |
| `--lora-modules` | LoRA 适配器映射 | `name=path` 格式 |

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 文本请求
response = client.chat.completions.create(
    model="model",  # 合并模式用任意值，LoRA 模式用 adapter 名称
    messages=[{"role": "user", "content": "Hello"}],
)
```

## 参考

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora.html)
- [vLLM OpenAI Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [serve/README.md](../serve/README.md) — 本项目部署脚本详细文档
