# Data

CIFAR 数据集下载、转换为 LLamaFactory 多模态训练格式。

## 数据格式

采用 **ShareGPT 多模态格式**（`<image>` 占位符 + `images` 数组），供 LLamaFactory 直接训练：

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Classify this image into one of these categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Answer with only the class name."
      },
      { "role": "assistant", "content": "airplane" }
    ],
    "images": ["data/images/cifar10/train/0_airplane_00001.png"]
  }
]
```

关键规则：
- `<image>` 标记数量与 `images` 数组长度严格一致
- 图片路径相对于 LLamaFactory 根目录（`dataset_info.json` 中配置）
- Qwen3.5-2B 接受图像+文本输入，直接使用原始图片即可

## 数据目录结构

```
data/
├── raw/                    # torchvision 下载的原始 CIFAR 文件
│   ├── cifar-10-batches-py/
│   └── cifar-100-python/
├── images/                 # 提取的 PNG 图片
│   ├── cifar10/
│   │   ├── class_mapping.json
│   │   ├── train/          # 0_airplane_00000.png, ...
│   │   └── test/           # 0_airplane_00000.png, ...
│   └── cifar100/
│       ├── class_mapping.json
│       ├── train/
│       └── test/
└── processed/              # ShareGPT JSON 训练数据
    ├── cifar10_train.json
    ├── cifar10_test.json
    ├── cifar100_train.json
    └── cifar100_test.json
```

## 脚本

### 1. `download_cifar.py` — 下载并保存图片

```bash
# 下载 CIFAR10（全部图片）
python data/download_cifar.py --dataset cifar10

# 下载 CIFAR100，每个类别只取 200 张（快速实验）
python data/download_cifar.py --dataset cifar100 --subset 200

# 同时下载两个数据集
python data/download_cifar.py --dataset all --subset 500
```

### 2. `build_instructions.py` — 构建训练 JSON

```bash
# 构建 CIFAR10 训练数据
python data/build_instructions.py --dataset cifar10

# 构建 CIFAR100，限制训练集 5000 条
python data/build_instructions.py --dataset cifar100 --max-train 5000

# 全部构建
python data/build_instructions.py --dataset all
```

## 运行顺序

```bash
# 先下载图片，再构建 JSON
python data/download_cifar.py --dataset cifar10 --subset 200
python data/build_instructions.py --dataset cifar10
```
