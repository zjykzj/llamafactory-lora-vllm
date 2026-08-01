# 提示词设计策略

> 提示词不是无关紧要的包装纸——在多模态模型训练中，措辞差异可导致高达 29% 的性能差距。

## 核心原则

### 原则 1：训练阶段使用多样化变体

**为什么**：[Template Matters (Wang et al., ICLR 2025)](https://ar5iv.labs.arxiv.org/html/2412.08307) 测试了 8 个开源多模态模型，发现：

- 同一张图、同一个问题，换一种问法，模型性能差距最高可达 **29%**
- 用多样化模板增强训练数据后，模型性能**超越了用 75 倍数据量但只有单一模板训练的模型**
- 模板多样性是数据效率极高的杠杆：改措辞几乎零成本，收益堪比数倍数据

[MSTaR (Liu et al., ICML 2025)](https://arxiv.org/abs/2412.17451) 将 Prompt Variation 列为自进化训练的三大支柱之一。

**本项目做法**：每个数据集 3~4 个语义等价的变体，训练时随机选取。

### 原则 2：评估提示词必须是训练变体之一

**为什么**：[Shiono et al. (2024)](https://ar5iv.labs.arxiv.org/html/2512.23572) 发现视觉指令微调后，VLM 的指令遵循能力会退化。但如果训练数据中**显式包含输出格式约束**（如 "Answer with only the class name."），退化就能被抑制。

eval prompt 与训练不一致会导致：
- 模型在训练时学到的"看到 X 句式 → 产出 Y 格式"映射失效
- 输出格式不受控（模型可能输出完整句子而非类名）

**本项目做法**：eval prompt 始终等于训练变体中的第一个。

### 原则 3：统一输出格式约束

所有变体必须以相同的输出格式约束结尾（如 `Answer with only the class name.`）。[Shiono et al.] 证明这能显著抑制指令遵循能力的退化。

## CIFAR10 vs CIFAR100：两类策略

两类数据集的提示词策略不同，核心差异在于**类别数量**：

| 维度 | CIFAR10 | CIFAR100 |
|------|---------|----------|
| 类别数 | 10 | 100 |
| 类别列表长度 | ~80 chars | ~1200 chars (~800 tokens) |
| 训练时是否包含类别列表 | ✅ 是（token 成本可忽略） | ❌ 否（每样本省 ~800 tokens） |
| 评估时是否包含类别列表 | ✅ 是（约束输出空间） | ❌ 否（模型从训练中学会类名） |

**CIFAR10**：类别列表短，包含在 prompt 中可作为输出空间约束，性价比高。

**CIFAR100**：类别列表长（每个训练样本重复一份 → 50000 × 800 = 4000 万 wasted tokens）。模型通过 assistant 回复学会 100 个类名，不需要在 prompt 中重复列举。

## 当前提示词

### CIFAR10

**训练**（4 个变体，随机选取）：

```
1. <image>Classify this image into one of these categories: {10类}. Answer with only the class name.
2. <image>What object is shown in this image? Choose from: {10类}. Answer with only the class name.
3. <image>Identify the main object in this picture. Options: {10类}. Answer with only the class name.
4. <image>Which category does this image belong to? Options: {10类}. Answer with only the class name.
```

**评估**（= 训练变体 1，去掉 `<image>` 因为图片作为 `image_url` 单独传入）：

```
Classify this image into one of these categories: {10类}. Answer with only the class name.
```

### CIFAR100

**训练**（3 个变体，随机选取，不含类别列表）：

```
1. <image>Classify this image. Answer with only the class name.
2. <image>What object is shown in this image? Answer with only the class name.
3. <image>Identify the main object in this picture. Answer with only the class name.
```

**评估**（= 训练变体 1）：

```
Classify this image. Answer with only the class name.
```

## 提示词变体的设计约束

本项目的提示词变体遵循以下约束：

1. **语义等价**：所有变体表达相同的任务（图片分类），不引入额外的推理要求
2. **输出格式统一**：所有变体以 `Answer with only the class name.` 结尾
3. **无元信息泄露**：不包含数据集名称（CIFAR-10 / CIFAR-100），这些在实际部署中不存在
4. **minimal 原则**：除类别列表（如有）外不添加冗余信息——CIFAR 是简单分类任务，不需要 CoT 或 few-shot 示例

## 参考

- [Template Matters: Understanding the Role of Instruction Templates in Multimodal Language Model Evaluation and Training](https://ar5iv.labs.arxiv.org/html/2412.08307) — Wang et al., ICLR 2025
- [Multi-modal Preference Alignment Remedies Degradation of Visual Instruction Tuning on Language Models](https://aclanthology.org/2024.acl-long.765/) — Li et al., ACL 2024
- [Diving into Self-Evolving Training for Multimodal Reasoning](https://arxiv.org/abs/2412.17451) — Liu et al., ICML 2025
- [Instruction-Following Evaluation of Large Vision-Language Models](https://ar5iv.labs.arxiv.org/html/2512.23572) — Shiono et al., 2024
