# Serve — 模型部署

## 部署方式概览

| 方式 | 后端 | 需要合并？ | 并发 | 适用场景 |
|------|------|-----------|------|---------|
| vLLM + LoRA | vLLM 原生 | ❌ 不需要 | ✅ 高 | **推荐**：生产部署，多任务切换 |
| vLLM + 合并模型 | vLLM 原生 | ✅ 需要 | ✅ 高 | 单任务，追求极致稳定 |
| LLaMA Factory API | llamafactory | ❌ 不需要 | ❌ 低 | 快速调试，单请求验证 |

> **三者都提供 OpenAI 兼容的 `/v1/chat/completions` 接口**，eval 脚本可通用。

## ModelScope 用户

如果模型是从 ModelScope 下载的（而非 HuggingFace），设置环境变量：

```bash
export USE_MODELSCOPE_HUB=1
```

然后再执行 `bash serve/serve.sh ...`。此变量会让底层 `transformers` / `vLLM` 从 ModelScope 而非 HuggingFace Hub 拉取模型。

---

## 方式一：vLLM + LoRA 适配器（推荐）

vLLM 原生支持直接加载 LoRA 适配器，**无需合并模型**，启动最快。

### 启动

```bash
# 默认 cifar10
bash serve/serve.sh --lora models/lora/cifar10

# 指定基座模型（通常自动从 adapter_config.json 读取）
bash serve/serve.sh --lora models/lora/cifar10 --base-model /path/to/Qwen3.5-2B

# cifar100
bash serve/serve.sh --lora models/lora/cifar100

# 自定义 host/port
bash serve/serve.sh --lora models/lora/cifar10 --port 8080
```

### 请求时指定 adapter

vLLM 根据请求中的 `model` 字段路由到对应 adapter：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# model 名对应 --lora 路径的目录名（cifar10 / cifar100）
response = client.chat.completions.create(
    model="cifar10",
    messages=[{"role": "user", "content": "..."}],
)
```

### 同时加载多个 adapter

vLLM 支持同时加载多个 LoRA adapter，请求时按 `model` 名动态切换：

```bash
vllm serve /path/to/Qwen3.5-2B \
    --enable-lora \
    --lora-modules cifar10=models/lora/cifar10 cifar100=models/lora/cifar100 \
    --host 0.0.0.0 --port 8000 --trust-remote-code
```

---

## 方式二：vLLM + 合并模型

先合并 LoRA 到基座模型，再部署。适合不需要切换任务的场景。

### 1. 合并

```bash
llamafactory-cli export configs/cifar10_merge.yaml
```

合并后的模型输出到 `models/merged/cifar10/`。

### 2. 启动

```bash
# 默认 cifar10 合并模型
bash serve/serve.sh

# 指定模型
bash serve/serve.sh --model models/merged/cifar100

# 模型名自动取目录 basename（如 cifar100），eval 时用：
#   python eval/eval_cifar100.py --model cifar100

# 等同于直接调用 vLLM
vllm serve models/merged/cifar10 \
    --served-model-name cifar10 \
    --host 0.0.0.0 --port 8000 --trust-remote-code
```

---

## 方式三：LLaMA Factory API

一条命令启动，适合快速调试。**不支持并发**，生产建议用 vLLM。

```bash
llamafactory-cli api configs/cifar10_infer.yaml
```

配置中的 `infer_backend` 可选择后端：

| infer_backend | 说明 |
|---------------|------|
| `huggingface` | 标准 transformers 推理 |
| `vllm` | LLaMA Factory 代理 vLLM 引擎 |
| `sglang` | SGLang 引擎 |

同样提供 `http://localhost:8000/v1` 接口，eval 脚本可直接使用。

---

## 环境变量

所有命令行参数均可通过环境变量设置（命令行参数优先级更高）：

| 环境变量 | 对应参数 | 默认值 |
|---------|---------|--------|
| `SERVE_HOST` | `--host` | `0.0.0.0` |
| `SERVE_PORT` | `--port` | `8000` |
| `SERVE_MODEL` | `--model` | `models/merged/cifar10` |
| `SERVE_ADAPTER` | `--lora` | - |
| `SERVE_BASE_MODEL` | `--base-model` | 自动从 adapter_config.json 读取 |
| `SERVE_GPU_MEMORY` | `--gpu-memory` | `0.9` |
| `SERVE_API_KEY` | `--api-key` | `not-needed` |
| `USE_MODELSCOPE_HUB` | — | `0` | 设为 `1` 从 ModelScope 加载模型 |

## 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 模型列表
curl http://localhost:8000/v1/models

# 推理测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cifar10",
    "messages": [{"role": "user", "content": "Say hello in one word."}]
  }'
```

## 常见问题

**端口被占用？**
```bash
bash serve/serve.sh --port 8080
```

**模型目录未找到（合并模式）？**
```bash
# 先合并 LoRA
llamafactory-cli export configs/cifar10_merge.yaml
# 或直接使用 LoRA 模式（跳过合并）
bash serve/serve.sh --lora models/lora/cifar10
```

**显存不足？**
```bash
bash serve/serve.sh --gpu-memory 0.5
```

**vLLM 未安装？**
```bash
pip install vllm
```
