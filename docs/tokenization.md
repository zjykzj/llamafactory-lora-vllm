# Tokenization：从文本到分词

> Tokenization 是大模型输入输出的第一道闸门——它决定了模型"看到"什么，也决定了模型能"说出"什么。

本文先讲实践（如何用 `AutoTokenizer` 计算 token 数），再讲原理（BPE 等分词算法的底层实现）。

## 为什么 Tokenization 重要

所有 LLM 的输入都不是原始文本，而是整数序列（token ids）。Tokenizer 负责这个转换：

```
文本 "Hello, world!"  →  Tokenizer  →  [15496, 11, 995, 0]
图像 (512×512 pixels)  →  Vision Encoder  →  [visual_token_1, visual_token_2, ...]
```

Tokenization 直接影响：
- **上下文长度**：prompt 太长 → 超 context window → 截断
- **推理成本**：每个 token 都要做一次前向传播，token 数 = 推理耗时
- **多语言公平性**：同一个词，中文 1 token，英文可能 3~5 tokens
- **任务精度**：类别名 "automobile" 被切成 2 个 token → 模型需要分两步预测

## 实践：AutoTokenizer 使用

### 加载 Tokenizer

```python
from transformers import AutoTokenizer

# 从 ModelScope 加载（本项目默认）
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.5-2B",
    trust_remote_code=True
)

# 从本地缓存加载
tokenizer = AutoTokenizer.from_pretrained(
    "/root/.cache/modelscope/Qwen/Qwen3.5-2B",
    trust_remote_code=True
)
```

`trust_remote_code=True` 对 Qwen 系列是**必需的**——Qwen 的 tokenizer 包含自定义逻辑（如 visual token 处理、chat template）。

### 计算 Token 数

```python
# 方法 1：encode 后看长度
text = "What is this?"
tokens = tokenizer.encode(text)
print(len(tokens))  # → 4

# 方法 2：直接 tokenize 看详情
tokens = tokenizer.tokenize(text)
print(tokens)  # → ['What', 'Ġis', 'Ġthis', '?']

# 方法 3：用 tokenizer 的返回值
encoded = tokenizer(text, return_tensors="pt")
print(encoded["input_ids"].shape)  # → torch.Size([1, 4])

# 方法 4：apply_chat_template（含 messages 格式）
messages = [
    {"role": "user", "content": "What is this?"}
]
tokenized = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
)
print(tokenized.shape)  # 包含了 chat template 特殊 token 后的总 token 数
```

### 查看 Token ↔ ID 映射

```python
# 单个 token ↔ id
print(tokenizer.encode("cat"))        # → [8682]  (单个 ID)
print(tokenizer.decode([8682]))       # → "cat"

# 查看 vocab 大小
print(tokenizer.vocab_size)           # Qwen3.5 → 248,000 (实际约 152K + special)

# 多 token 词
print(tokenizer.tokenize("automobile"))  # → ['autom', 'obile']  被切成 2 个
print(tokenizer.encode("automobile"))    # → [12048, 4647]
```

### 多模态 token 计算（图像）

```python
# Qwen3.5 视觉 token 数的计算公式
def count_visual_tokens(image_width, image_height, patch_size=16, merge_size=2):
    """
    Qwen3.5 Vision Encoder 的 token 计算：
    1. Patch 切分：图像 → (H/patch) × (W/patch) 个 patch
    2. Spatial Merge：2×2 合并，减少 token 数
    """
    h_patches = image_height // patch_size
    w_patches = image_width // patch_size
    # 经过 2×2 spatial merge
    merged_h = h_patches // merge_size
    merged_w = w_patches // merge_size
    return merged_h * merged_w

# CIFAR 32×32 → 极小
print(count_visual_tokens(32, 32))   # → 1   (32/16=2, 2x2 merge → 1 token)

# CIFAR 经过 image_max_pixels 限制后
# 如果 image_max_pixels=262144 (512×512)
print(count_visual_tokens(512, 512)) # → 256  (512/16=32, 32×32=1024, merge后=256)

# 如果 image_max_pixels=65536 (256×256)
print(count_visual_tokens(256, 256)) # → 64   (256/16=16, 16×16=256, merge后=64)
```

**CIFAR 分类实际 token 构成**：

```
[<|im_start|> system ...]   ← 系统提示 (~15 tokens)
[<vision_start>]            ← 视觉开始标记
[image_patch_1, ...]        ← 图片 patch tokens (1~256 个，取决于分辨率)
[<vision_end>]              ← 视觉结束标记
[What is this?]             ← 用户问题 (~4 tokens)
[<|im_end|>]                ← 结束
[<|im_start|> assistant]    ← assistant 前缀
[automobile]                ← 模型输出 (1~3 tokens)
```

### 从 token 数诊断问题

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)

# CIFAR10 所有类别名的 token 长度
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

for cls in classes:
    tokens = tokenizer.encode(cls)
    print(f"{cls:>12}: {len(tokens)} token(s) → {tokens} → {tokenizer.tokenize(cls)}")
```

输出示例：

```
    airplane: 1 token(s) → [32062] → ['airplane']
  automobile: 2 token(s) → [12048, 4647] → ['autom', 'obile']    ← 注意！
       bird: 1 token(s) → [28364] → ['bird']
        cat: 1 token(s) → [8682] → ['cat']
       deer: 1 token(s) → [51368] → ['deer']
        dog: 1 token(s) → [11463] → ['dog']
       frog: 1 token(s) → [38694] → ['frog']
      horse: 1 token(s) → [31811] → ['horse']
       ship: 1 token(s) → [11729] → ['ship']
      truck: 1 token(s) → [10382] → ['truck']
```

**"automobile" 是唯一被切成 2 个 token 的 CIFAR10 类别**。这意味着模型预测它需要两步（先猜 "autom"，再猜 "obile"），比一步猜出 "airplane" 多一次出错机会。这是 tokenizer 对分类精度的隐性影响。

## 原理：Tokenization 算法

### 为什么需要 Tokenization

直接按字符编码有两个问题：

1. **字符数 ≠ 语义单位**："cat" 是 3 个字符，但语义上是 1 个概念。按字符建模会浪费计算
2. **vocab 爆炸**：按词编码的话，英文有几十万词，中文有无限组合——vocab 会无限大

Tokenization 在两者之间折衷：**子词（subword）级别的切分**——常用词保持完整，罕见词拆分成可组装的碎片。

### BPE（Byte Pair Encoding）— 当前主流

GPT 系列、Llama 系列、Qwen 系列均使用 BPE 或其变体。

#### 核心思想

BPE 的做法像压缩算法：从字符开始，反复将最常见的相邻字符对合并成一个新 token。

#### 算法步骤

**Step 1：初始化**

将每个字符作为独立 token，语料中每个词末尾加一个特殊标记 `</w>`（表示词边界）：

```
"low"  →  l o w </w>
"lower" → l o w e r </w>
```

**Step 2：统计相邻 pair 频率**

```
(l, o):  2 次
(o, w):  2 次
(w, </w>): 1 次
(w, e):  1 次
...
```

**Step 3：合并频率最高的 pair**

假设 (l, o) 频率最高 → 合并为 "lo"：

```
"low"  →  lo w </w>
"lower" → lo w e r </w>
```

**Step 4：重复，直到 vocab 达到目标大小**

```
迭代 1:  (l, o)    → "lo"
迭代 2:  (lo, w)   → "low"
迭代 3:  (w, e)    → "we"
迭代 4:  (e, r)    → "er"
迭代 5:  (er, </w>) → "er</w>"
...
```

最终 vocab 包含从字符到完整词的各种粒度：
```
[..., "l", "o", "w", "lo", "low", "e", "r", "er", "er</w>", ...]
```

#### 编码过程

给定训练好的 vocab，编码新词：

```
输入 "lowest"（训练时未见过）
→ 从字符开始：l o w e s t
→ 应用合并规则（按训练时的优先级）：
  l + o → lo
  lo + w → low
  e + s → es
  es + t → est
→ 输出：[low, est]
```

#### 解码过程

```
[low, est] → "low" + "est" → "lowest"
```

解码极其简单——只需拼接。这也是 BPE 的一个优势：**解码完全确定，不需要任何搜索或概率计算**。

### BPE 变体对比

| 算法 | 核心合并策略 | 代表模型 | 特点 |
|------|------------|---------|------|
| **BPE** | 基于频率合并 pair | GPT-2/3/4, Llama | 贪婪合并，频率驱动 |
| **BBPE** (Byte-level BPE) | BPE 但基础单元是字节（256 种） | GPT-2+, Llama, Qwen | 天然支持任何语言（不会出现 UNK） |
| **WordPiece** | 基于概率增益合并 pair | BERT, DistilBERT | BPE 但用 likelihood 而非频率选 pair |
| **Unigram** | 基于 loss 剪枝（先过生成再缩） | T5, XLNet, mBART | 与 BPE 相反——从大 vocab 开始逐步剪枝 |
| **SentencePiece** | 库，不是算法；支持 BPE/Unigram | Llama, T5, Qwen | 将空格视为普通字符，语言无关 |

### BBPE：Qwen3.5 的 Tokenizer

Qwen3.5 使用的是 **Byte-level BPE (BBPE)**，通过 SentencePiece 库实现。

**BBPE 的关键创新**：基础单元不是 Unicode 字符，而是**字节（byte，0~255）**。

```
传统 BPE  的基础单元：   a, b, c, ..., 你, 好, ..., 😊, ...   (Unicode chars, ~150K)
BBPE 的基础单元：        0x00, 0x01, ..., 0xFF                 (bytes, 256)
```

**优势**：
1. **不会出现 [UNK]**：任何 Unicode 字符都可以分解为 1~4 个字节序列。模型永远不会遇到"不认识的字"
2. **跨语言统一**：所有语言在字节层面是统一的，不存在中文分词、日文分词的特殊处理
3. **emoji 天然支持**：😊 → UTF-8 bytes → BBPE tokens

**代价**：
- 罕见字符（如特殊 emoji）会被切成多个 byte token，序列变长
- 常见中文字可能比专用中文 tokenizer 用更多 tokens

### 训练 BPE Tokenizer

```python
# 概念代码——实际使用 tokenizers 库
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. 初始化
tokenizer = Tokenizer(models.BPE())

# 2. 配置预分词器（BBPE 从字节开始，跳过这步）
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 3. 训练
trainer = trainers.BpeTrainer(
    vocab_size=152000,         # Qwen3.5 的 vocab 大小
    special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
    min_frequency=2,           # pair 至少出现 2 次才能合并
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

Qwen3.5 的 tokenizer 包含约 **152,000 个 BPE tokens** + 约 **96,000 个 special/reserved tokens** = 约 **248,000 总 vocab**。

那 96K 的 reserved tokens 用于：
- Chat template 控制 token（`<|im_start|>`, `<|im_end|>` 等）
- 视觉 token（`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>` 等）
- 多语言扩展预留
- 未来功能预留

### Tokenizer 的生命周期

```
训练阶段（一次性，离线完成）：
  大规模语料 → 统计 byte pair 频率 → 迭代合并 → 生成 vocab.json + merges.txt

使用阶段（每次推理都在做）：
  输入文本 → 按合并规则编码 → token IDs → 模型推理
  模型输出 token IDs → 查 vocab 表 → 拼接 → 文本
```

你加载 `AutoTokenizer.from_pretrained()` 时，实际上加载的是**已经训练好的** `vocab.json` 和 `merges.txt`（或 `tokenizer.model`）。不会重新训练——这需要数 TB 语料。

## Special Tokens 和 Chat Template

### 特殊 Token 类型

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)

print(tokenizer.bos_token)     # <|im_start|>    序列开始
print(tokenizer.eos_token)     # <|im_end|>      序列结束
print(tokenizer.pad_token)     # <|endoftext|>   填充
print(tokenizer.unk_token)     # None            BBPE 不需要 UNK
```

Qwen3.5 的 chat template 结构：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is this?<|im_end|>
<|im_start|>assistant
automobile<|im_end|>
```

### apply_chat_template 的作用

```python
messages = [{"role": "user", "content": "What is this?"}]

# 自动拼接 chat template special tokens
tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt")
# 生成的 ids 自动包含了 <|im_start|>user\n...<|im_end|> 等控制 token
```

这保证了训练和推理时 token 格式完全一致——如果自己手动拼接，很容易漏掉或多加空格/换行，导致格式漂移。

## 总结

| 概念 | 一句话 |
|------|--------|
| Tokenization | 将文本/图像转换为整数 ID 序列 |
| BPE | 从字符开始，反复合并最常见的相邻 pair |
| BBPE | BPE 但基础单元是字节——永不出现 UNK |
| SentencePiece | 实现库，支持 BPE/Unigram，语言无关 |
| `AutoTokenizer` | HuggingFace 的统一加载接口 |
| `encode()` / `decode()` | 文本 ↔ token IDs 的双向转换 |
| `apply_chat_template()` | 自动拼接 chat format 的特殊 token |
| 视觉 token | Patch 切分 → 2×2 merge → 1 token/4 pixels（以 16px patch 计） |

## References

- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — BPE 原始论文
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer](https://arxiv.org/abs/1808.06226)
- [Byte Pair Encoding (HuggingFace NLP Course)](https://huggingface.co/learn/nlp-course/chapter6/5)
- [HuggingFace Tokenizers Library](https://github.com/huggingface/tokenizers) — Rust 实现的高性能 tokenizer
- [Qwen3.5 Technical Report](https://arxiv.org/abs/2602.19444) — Vocab 大小和 tokenizer 细节
