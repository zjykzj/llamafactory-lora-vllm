# 大模型中的 Cross-Entropy Loss：从判别到生成

> 同一个 Cross-Entropy Loss，在传统视觉分类和自回归大模型中扮演着截然不同的角色。

本文从 CE Loss 的数学定义出发，分析它在 Qwen3.5 预训练、多模态对齐、指令微调、以及 CIFAR 分类 LoRA 微调中的统一形式，并对比两种分类范式的本质差异。

## CE Loss 的两种面孔

Cross-Entropy 的数学定义完全一样：

$$H(p, q) = -\sum_{i} p(i) \cdot \log q(i)$$

但**输出空间 $i$** 的决定方式截然不同：

### 面孔一：判别式分类（传统视觉）

模型最后一层是一个固定维度的分类头（如 Linear(768, 1000)），输出 $K$ 个 logits，每个对应一个**固定的类别**：

$$\mathcal{L}_{\text{disc}} = -\log \frac{e^{z_y}}{\sum_{k=1}^{K} e^{z_k}}, \quad K = 10 \text{ 或 } 1000$$

- 预测 "飞机" → 输出空间只有 10/100/1000 个选项
- "飞机" 和 "狗" 在这个空间里是**互斥**的
- 分类头权重 $W \in \mathbb{R}^{d \times K}$ 中的每一列都可以理解为"类别 k 的模板向量"

### 面孔二：生成式分类（自回归 LM）

模型的最后一层是 **LM Head**（vocab projection），输出 **248K 维**的 logits，每个 token 一次预测：

$$\mathcal{L}_{\text{gen}} = -\sum_{t \in \text{response}} \log P(y_t \mid \text{image}, \text{instruction}, y_{<t})$$

- 模型从 248K 个 token 里选，而不是 10/100 个类别
- 分类是**间接达成的**——通过描述图像内容、说出类别名称
- "类别名称"只是 248K 个候选 token 中的几个

**核心区别**：判别式分类是「这个图属于哪一类」的直接映射；生成式分类是「你看到的是什么，用语言说出来」——模型需要理解类别名称的语义才能正确输出。

## Qwen3.5 全生命周期的 CE Loss

Qwen3.5 从出生到做 CIFAR 分类，从头至尾只用一个 loss，但每个阶段的输入不同：

```
阶段 1：文本预训练（数万亿 token 文本）
─────────────────────────────────────────
输入：  [今天, 天气, 真] → 预测：好
输入：  [太阳, 从, 哪边, 升, 起] → 预测：？
Loss：  CE(next token | previous tokens)
目的：  学语言的统计规律和世界知识


阶段 2：多模态预训练（图文对数据）
─────────────────────────────────────────
输入：  [<img: 一张猫的照片>] + "这是一只" → 预测：猫
输入：  [<img: 一辆红色跑车>] + "图中的车是" → 预测：红
Loss：  CE(next token | image + text)
目的：  建立视觉特征与语言概念的关联


阶段 3：SFT 指令微调（对话数据）
─────────────────────────────────────────
输入：  [<img: 一只狗>] + "图中是什么动物？" → 预测：这是一只金毛犬。
Loss：  CE(next token | image + instruction + previous response tokens)
目的：  学会遵循指令格式，按人类期望的方式作答


阶段 4：CIFAR LoRA 分类微调（分类数据）
─────────────────────────────────────────
输入：  [<img: CIFAR automobile>] + "What is this?" → 预测：automobile
Loss：  CE(next token | image + instruction)
目的：  在特定分类任务上精确调整
```

### 同一个 loss 函数，四个不同任务

关键在于 **CE Loss 不关心任务是什么**——它只关心"你在给定上下文后，猜对下一个 token 了吗"：

| 阶段 | 看似在做什么 | 实际在做什么（从 CE 视角） |
|------|-----------|------------------------|
| 文本预训练 | 学语言 | 猜下一个词 |
| 多模态预训练 | 学看图 | 图中有什么，就猜什么词 |
| SFT | 学格式 | 按指令格式猜 answer |
| CIFAR 分类 | 学分类 | 图中是哪类，就猜哪个类名 |

### 为什么这种统一有效

**世界知识已经被编码在语言里了**。

模型在文本预训练阶段见过 "automobile is a four-wheeled vehicle"、"cat is a feline animal" 这样的描述。多模态预训练时见过猫的图片和"cat"这个词的关联。当它看到 CIFAR 的 automobile 图片时，它**不是从零学"什么是汽车"**，而是在激活和微调预训练阶段已经建立的视觉-语义关联。

这就是 Qwen3.5 zero-shot 就能拿 96.12% 的原因——没有 CIFAR 微调，它已经知道 automobile 长什么样了。

## 两种范式的深度对比

### 从模型架构角度

```
判别式分类 (ResNet/ViT):
  Image → Encoder → [CLS] hidden state → Linear(768, 10) → CE(10 logits, label)

生成式分类 (Qwen3.5):
  Image → Vision Encoder → LM Backbone → LM Head(768, 248K) → CE(248K logits, token)
  每个 response token 独立算一次 CE
```

| 维度 | 判别式 | 生成式 |
|------|--------|--------|
| 输出维度 | K 类（固定） | 248K vocab（固定） |
| 决策次数 | 1 次 | response 的每个 token 各 1 次 |
| 类别映射 | 线性层权重直接学习 "第 3 维 = cat" | 通过 tokenizer 间接映射 "cat" → [token_id] |
| 训练信号 | 仅分类标签 | 每一个 token 都是信号 |
| 泛化能力 | 仅限训练过的类 | **可泛化到未见过的类**（zero-shot） |
| 预训练知识利用 | 需从头训练分类头 | 预训练的 vocab embedding 直接复用 |

### 从分类精度角度

对 CIFAR 这类**类别名称本身就是自然语言词汇**的任务，生成式有独特的优势：

1. **类间语义关联被保留**："automobile" 和 "truck" 在 vocab 空间里是近邻（共享相似的上下文分布），而不是 10 维 one-hot 里两个无关的维度
2. **预训练 priors 提供强先验**：zero-shot 96% 已经很说明问题——判别式模型 zero-shot 是 10%
3. **一个模型同时做 10 类和 100 类，不换分类头**：这正是 LoRA 能跨任务共享 base model 的原因

### 劣势

1. **效率低**：每个 token 都要在 248K 维上做 softmax（实际有优化，但理论开销大于 10 维）
2. **tokenizer bias**："cat"是 1 个 token，"automobile" 可能是 2~3 个——长度不同的类别名预测难度不均
3. **可解释性弱**：没法直接看"模型的 embedding 空间中 automobile 和 truck 的距离"，要看各个类名的 token 概率

## 视觉生成式分类：正在形成的范式

这个范式不是 Qwen3.5 独有的。以下模型都用了类似的"视觉输入 + 文本输出 + CE Loss"方案：

| 模型 | 发布 | 方案 |
|------|------|------|
| CLIP | 2021 | 对比学习（非生成），但开创了"视觉+文本统一空间" |
| Flamingo | 2022 | 视觉编码 + LM backbone，CE 生成文本描述 |
| BLIP-2 | 2023 | Q-Former 桥接视觉编码器和 LLM，CE 生成 |
| LLaVA 1.5 | 2023 | 简单线性投影 + LLM，CE 生成 |
| GPT-4V | 2023 | 原生多模态 LLM，统一 CE |
| Qwen2-VL | 2024 | 原生多模态 LLM，统一 CE |
| **Qwen3.5** | 2026 | 原生多模态 + Hybrid Attention，统一 CE |

趋势明显：**视觉理解正在被统一到语言生成范式下**。分类、检测、VQA、captioning——所有视觉任务都可以用同一个模型、同一个 CE Loss、同一个 vocab 来解决，只需换 prompt。

## 对本项目 CIFAR 分类的启示

### 当前做法的问题

```python
# eval 脚本实际做的事
response = client.chat.completions.create(
    model="cifar10_qwen3.5-9B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "What is this?"},
    ]}],
)

predicted = response["choices"][0]["message"]["content"].strip()  # "automobile"
accuracy += (predicted in CLASS_NAMES)  # 类别名匹配
```

精度是从**生成的完整文本**中事后提取的。如果模型输出 "This is an automobile." 而代码只匹配 "automobile"，就会漏判。

### 已知的优化方向

1. **Prompt 工程**：约束输出格式（"Answer with exactly one word"），减少无关 token 的生成
2. **Log-prob 直接比较**：不生成文本，而是计算每个类别名称的 log-prob，选最大者——本质上是把生成式模型"回退"到判别式用法
3. **JSON 约束输出**（guided decoding）：限制模型只能输出预定义的类别列表

方法 2 是理论上最合理的做法：

```python
# 不生成文本，直接比较 log-prob
for class_name in ["airplane", "automobile", "bird", ...]:
    logp = model.log_prob(class_name | image, "What is this?")
predicted = argmax(logp)
```

这利用了 CE Loss 的核心能力——概率估计——而不是依赖偶然的文本匹配。

## References

- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — "所有任务统一为生成任务"的原始论证
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — 视觉指令微调
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
