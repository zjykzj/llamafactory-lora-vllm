# 多模态数据集格式

LLamaFactory 训练 Qwen3.5-2B 等多模态模型时，使用 **ShareGPT 格式 + `<image>` 占位符** 的数据格式。

## 数据格式

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Classify this image into one of these categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."
      },
      { "role": "assistant", "content": "airplane" }
    ],
    "images": ["data/images/cifar10/train/0_airplane_00001.png"]
  }
]
```

## 关键规则

1. **`<image>` 与 `images` 一一对应**：文本中的 `<image>` 数量必须等于 `images` 数组长度
2. **支持多图**：多张图片时使用 `<image1>`、`<image2>` 区分
3. **图片路径**：相对于 LLaMA-Factory 根目录的相对路径
4. **支持格式**：JPG、PNG、BMP

## 多图示例

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image1> and <image2>: which one is a cat?"
    },
    { "role": "assistant", "content": "<image1> is a cat, <image2> is a dog." }
  ],
  "images": ["data/images/cat.png", "data/images/dog.png"]
}
```

## dataset_info.json 注册

```json
{
  "cifar10_train": {
    "file_name": "data/processed/cifar10_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  }
}
```

## 训练 YAML 配置

```yaml
### Dataset
dataset: cifar10_train
template: qwen3_vl
cutoff_len: 2048

### Model
model_name_or_path: Qwen/Qwen3.5-2B
image_max_pixels: 262144
trust_remote_code: true
```

## 参考

- LLamaFactory `data/README.md` — MLLM 数据集说明
- LLamaFactory `examples/train_lora/` — 多模态 LoRA 训练示例
