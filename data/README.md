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
| 名称 | TODO(数据与实验)：待定稿 |
| 来源 | UCI Machine Learning Repository |
| 数据集编号 | TODO：见 `scripts/download_data.py` 中的 `DATASET_ID` |
| 许可 / 使用条款 | 公开数据集，引用其官方页面 |
| 样本数 | TODO |
| 特征数 | TODO |
| 标签列 | `default`（1 = 违约） |
| 正负比 | TODO：违约样本占比 |
| 缺失情况 | TODO |

## 处理后文件

| 文件 | 生成脚本 | 说明 |
| --- | --- | --- |
| `data/processed/credit_default_clean.csv` | `src/data.py::build_dataset` | 去重、缺失值与异常值处理后的数据 |
| `data/processed/split_index.json` | `src/data.py::make_split` | 训练 / 验证 / 测试索引，保证划分可复现 |

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
