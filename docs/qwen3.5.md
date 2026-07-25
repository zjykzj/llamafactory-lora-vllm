# Qwen3.5

[Qwen3.5](https://modelscope.cn/organization/qwen) 是多模态视觉语言模型（VLM），支持图像+文本输入。本项目使用 0.8B 和 2B 两个规模进行 CIFAR 分类微调。

## Model Configuration

| Configuration | Qwen3.5-0.8B | Qwen3.5-2B |
|--------------|-------------|-----------|
| Hidden Size | 1024 | 2048 |
| Layers | 24 | 24 |
| Attention Heads | 8 | 8 |
| KV Heads | 2 | 2 |
| Intermediate Size | 3584 | 6144 |
| Max Context | 256K | 256K |
| Disk Size | 1.7 GB | 4.2 GB |

- 两者架构相同（24 层 Transformer），2B 通过将 hidden size 翻倍（1024 → 2048）增加参数量。
- 均使用 Grouped Query Attention（8 个 query head + 2 个 KV head）。
- 混合注意力设计：75% linear attention（Gated DeltaNet）+ 25% full attention，每 4 层切换一次。
- 最大上下文 256K tokens，词表大小 248,320。
