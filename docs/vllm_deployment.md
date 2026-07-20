# vLLM 部署

## 概述

[vLLM](https://github.com/vllm-project/vllm) 是一个高性能 LLM 推理引擎，核心特性：

- **PagedAttention**：高效管理 KV cache，提升吞吐量
- **Continuous batching**：动态合并请求，降低延迟
- **OpenAI 兼容 API**：可直接替换 OpenAI 客户端
- **多模态支持**：图片输入等

## 部署流程

### 1. 合并 LoRA 模型

vLLM 不支持直接加载 LoRA adapter，需要先合并到基座模型：

```bash
llamafactory-cli export -c configs/cifar10_merge.yaml
```

### 2. 启动服务

```bash
vllm serve models/merged/cifar10 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

或使用本项目的脚本：

```bash
bash serve/serve.sh models/merged/cifar10
```

### 3. 验证

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

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 文本请求
response = client.chat.completions.create(
    model="model",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## 参考

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM OpenAI Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
