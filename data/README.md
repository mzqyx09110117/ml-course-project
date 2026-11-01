# 数据说明

本目录只保留数据说明，**原始数据与中间数据不纳入版本库**（见根目录 `.gitignore`）。

## 目录约定

```text
data/
├── README.md        # 本文件（入库）
├── raw/             # 原始数据，由脚本下载，不入库
└── processed/       # 清洗后的数据，由脚本生成，不入库
```

## 获取方式

```bash
python scripts/download_data.py
```

脚本通过 UCI 官方客户端拉取数据并写入 `data/raw/credit_default.csv`。

## 数据集信息

| 项目 | 内容 |
| --- | --- |
| 名称 | Default of Credit Card Clients |
| 来源 | UCI Machine Learning Repository（id=350） |
| 数据集编号 | 350 |
| 许可 / 使用条款 | 公开数据集，引用其官方页面 |
| 样本数 | 原始 30,000 行；剔除 35 条完全重复后 29,965 行 |
| 特征数 | 原始 23 个字段；特征工程后 43 维（新增 20 个衍生特征） |
| 标签列 | `default`（1 = 违约） |
| 正负比 | 违约 22.13% : 未违约 77.87% |
| 缺失情况 | 无缺失值 |

已知数据质量问题（已在 `src/data.py::clean` 中处理）：

| 问题 | 数量 | 处理方式 |
| --- | --- | --- |
| `EDUCATION` 未文档化取值（0 / 5 / 6） | 345 处 | 归入"其他"（4） |
| `MARRIAGE` 未文档化取值（0） | 54 处 | 归入"其他"（3） |
| 账单金额为负 | 3,932 处 | 视为退款，保留（比率特征中做 `+1` 平移） |
| 完全重复记录 | 35 条 | 剔除 |

## 处理后文件

| 文件 | 生成脚本 | 说明 |
| --- | --- | --- |
| `data/processed/credit_default_clean.csv` | `src/data.py::clean` | 去重、未文档化类别归并后的数据 |
| `data/processed/split_index.json` | `src/data.py::make_split` | 分层 70/15/15 划分索引，保证划分可复现 |

## 校验

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/raw/credit_default.csv")
print(df.shape)
print(df.head())
print(df["default"].value_counts(normalize=True))
PY
```

期望输出与上表"样本数 / 特征数 / 正负比"一致；若不一致，说明数据源版本已变化，
需同步更新本文件与 `scripts/download_data.py`。

## 注意事项

1. 数据中存在类别不平衡，**评估时不得只看 Accuracy**，主指标为 PR-AUC / F1 / Recall。
2. 训练集与测试集必须分层划分（`stratify=y`），否则正负比会随机漂移。
3. 任何重采样（过采样 / 欠采样 / 合成样本）只能作用于训练集，测试集保持原始分布。
