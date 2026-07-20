# Serve

vLLM 推理部署。

## 前置条件

训练完成的 LoRA 模型已通过 `llamafactory-cli export` 合并为完整模型，放在 `models/merged/` 目录下。

## 启动服务

```bash
# 默认启动 CIFAR10 合并模型
bash serve/serve.sh

# 指定模型路径
bash serve/serve.sh models/merged/cifar100

# 自定义 host/port
HOST=0.0.0.0 PORT=8080 bash serve/serve.sh
```

## 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 模型列表
curl http://localhost:8000/v1/models
```

## API 调用

vLLM 提供 OpenAI 兼容接口，详见 `eval/` 目录中的评估脚本。
