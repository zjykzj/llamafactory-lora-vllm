# Docs

## 模型架构与理论

- **[Qwen3.5](qwen3.5.md)** — 模型家族、Hybrid Attention 架构（DeltaNet + GQA）、Vision Encoder、FP8 训练
- **[自回归语言模型架构](autoregressive_models.md)** — Decoder-only Transformer 原理、Causal Attention/RoPE/SwiGLU/KV Cache，对比 BERT/T5/Mamba/MoE
- **[扩散模型架构](diffusion_models.md)** — DDPM/DDIM 原理、U-Net → DiT 演进、视频扩散（Seedance 2.0）、对比 GAN/VAE/Autoregressive/Flow Matching
- **[CE Loss 在大模型中的角色](ce_loss.md)** — 判别式 vs 生成式 CE、Qwen3.5 全生命周期 loss 分析、为什么分类任务能用生成范式解决
- **[Tokenization：从文本到分词](tokenization.md)** — AutoTokenizer 实践、视觉 token 计算、BPE/BBPE 算法原理

## 微调

- **[LoRA / QLoRA](lora_qlora.md)** — LoRA 原理、Rank 深入理解、QLoRA 精度对比、分模型调参建议
- **[训练配置](training_config.md)** — 有效 batch size、分模型显存估算、参数方案、验证集、CIFAR10/100 差异

## 数据与部署

- **[多模态数据集格式](multimodal_dataset.md)** — LLaMA Factory ShareGPT + `<image>` 数据格式规范
- **[vLLM 部署](vllm_deployment.md)** — LoRA 直接部署 / 合并部署、场景调参指南（低显存/多卡/高并发）
