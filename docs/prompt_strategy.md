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

## CIFAR10 vs CIFAR100

两类数据集的提示词策略**相同**：训练和评估都包含完整的类别列表，约束模型的输出空间。

| 维度 | CIFAR10 | CIFAR100 |
|------|---------|----------|
| 类别数 | 10 | 100 |
| 类别列表长度 | ~80 chars | ~1200 chars |
| 训练时包含类别列表 | ✅ | ✅ |
| 评估时包含类别列表 | ✅ | ✅ |

CIFAR100 的类别列表虽然长（~800 tokens/样本），但实验表明它对精度有正面贡献——显式约束输出空间比"让模型自己回忆类名"更准确，尤其是对较小模型（0.8B）。

### 评估标签匹配

CIFAR100 有 9 个类名包含下划线（如 `pickup_truck`、`sweet_pepper`）。评估时使用 `normalize_label()` 函数将下划线、连字符、空格统一折叠为空格后再比较，避免模型输出 `"pickup truck"`（空格）与 ground truth `"pickup_truck"`（下划线）不匹配的假阴性。

## 提示词详细程度：什么时候该加细节

提示词不是越详细越好——**细节的投入应与输出的结构复杂度匹配**。

### 决策框架

```
输出格式复杂度 ↑  →  需要更多格式约束（字符数、分隔符、正则等）
领域知识稀有度 ↑  →  需要更多上下文提示（但不教模型"认识"世界）
输出空间大小   ↑  →  需要更多类别锚点（但如果太大则不值得逐条列举）
```

### 案例对比：CIFAR10 分类 vs CCPD 车牌识别

| 维度 | CIFAR10 分类 | CCPD 车牌识别 |
|------|-------------|-------------|
| 输出空间 | 10 个离散类名 | 组合爆炸（省×字母×数字，~10¹³ 种） |
| 输出结构 | 无结构，1-2 个 token | 强结构，固定 7 字符，首字符必为省份 |
| 领域知识 | 类名即常识 | 需要知道中国车牌规则 |
| prompt 作用 | 仅激活分类行为 | 约束输出格式 + 激活领域知识 |

**CCPD 提示词对比**：

```
# ❌ 太简略 — 模型可能输出英文、缺位、格式错误
"<image>Read the plate number."

# ✅ 适度 — 给出结构锚点，不堆砌细节
"<image>Read the Chinese license plate. The plate has 7 characters: "
"a Chinese province character + a letter + 5 digits/letters. "
"Output only the plate number, nothing else."

# ❌ 过度 — 列举 31 个省份、解释 GA36 标准，浪费 token
"<image>This is a Chinese blue-plate vehicle license plate image "
"captured by a surveillance camera. Chinese plates follow the GA36 "
"standard: the first character is a single Chinese character "
"representing one of 31 provinces (京津冀晋蒙辽吉黑...), ..."
```

### 原则

1. **格式约束 > 知识灌输**：告诉模型"输出应该长什么样"（7 字符、省份开头），比罗列领域知识更有效
2. **结构锚点 > 全集枚举**：给关键规则（"首字符为省份缩写"），不枚举全量候选（31 个省份名）
3. **Qwen3.5 已经有常识**：它知道什么是中国车牌、省份有哪些，你只需要引导它用正确的格式输出
4. **CIFAR 的简单性**：分类任务的输出空间已在 assistant 回复中定义，prompt 只需激活分类行为

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

**训练**（3 个变体，随机选取）：

```
1. <image>Classify this image into one of these categories: {100类}. Answer with only the class name.
2. <image>What object is shown in this image? Choose from: {100类}. Answer with only the class name.
3. <image>Identify the main object in this picture. Options: {100类}. Answer with only the class name.
```

**评估**（= 训练变体 1）：

```
Classify this image into one of these categories: {100类}. Answer with only the class name.
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
