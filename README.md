# 信用违约风险预测

基于 UCI Default of Credit Card Clients 的表格二分类项目：预测客户下月是否违约，重点处理类别不平衡（正类 22.13%）与误判代价不对称（FN:FP = 10:1）。

作者：祁震淼

## 数据

| 项目 | 内容 |
| --- | --- |
| 来源 | UCI Default of Credit Card Clients（id=350） |
| 规模 | 30,000 行 × 24 列，去重后 29,965 行 |
| 标签 | 下月是否违约（1 = 违约），正类 22.13% |
| 特征 | 23 个原始字段 + 20 个派生特征 = 43 维 |
| 划分 | 分层 70/15/15，20,975 / 4,495 / 4,495 |

## 方法

- 基线：逻辑回归、随机森林；主模型：LightGBM、XGBoost。
- 不平衡处理：不处理、class_weight、SMOTE（仅训练折内）。
- 概率校准：Platt、Isotonic（仅验证集拟合）。
- 决策：在验证集上按期望成本最小化选阈值，不固定 0.5。
- 解释：TreeSHAP 与错误分析。

## 主要结果

测试集，5 个随机种子均值：

| 配置 | AUC | AP | Brier | ECE | 期望成本 | 阈值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost / none | 0.7861 | 0.5605 | 0.1342 | 0.0142 | 3104.2 | 0.07 |
| LightGBM / none | 0.7834 | 0.5572 | 0.1352 | 0.0216 | 3147.6 | 0.074 |
| 随机森林 / none | 0.7721 | 0.5500 | 0.1371 | 0.0201 | 3248.0 | 0.076 |
| 逻辑回归 / none | 0.7674 | 0.5358 | 0.1378 | 0.0210 | 3340.0 | 0.09 |

- 代价敏感阈值把期望成本降低 24.6%–53.0%，是收益最大的环节。
- class_weight 与 SMOTE 未提升判别力，却显著破坏概率校准。
- 校准改善 ECE，但基本不改变期望成本。

10 种配置、校准对比、消融与稳定性结果见 `results/tables/`。

## 目录

```text
src/       数据处理、模型、训练与评估
scripts/   数据下载与实验脚本
results/   表格、图与逐次运行指标
tests/     冒烟测试
data/      数据说明（原始数据不入库）
```

## 运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_experiments.py
python scripts/extra_calibration_imbalance.py
python scripts/extra_error_analysis.py
```

`--quick` 只跑 41 次单种子流程，用于自检；完整实验 81 次，CPU 约 8.5 分钟。环境版本见 `requirements.txt`，运行日志见 `_runs_log.md`。

## 复现

- 数据划分固定为 `SEED=42`，索引写入 `data/processed/split_index.json`。
- 校准器只在验证集拟合，阈值只在验证集选择，测试集只评估一次。
- 所有随机种子与成本参数集中在 `src/config.py`。

## 许可

MIT License.
