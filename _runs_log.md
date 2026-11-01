# 实验运行日志

## 机器与环境

- CPU：13th Gen Intel Core i5-13500H（16 线程），14 GiB RAM，无 GPU。
- 运行期间机器上另有其他任务占用约 7 个核心；本实验统一 `n_jobs=6`，避免抢满 CPU 导致耗时不可比。
- Python 3.13.9；lightgbm 4.7.0，xgboost 3.4.1，scikit-learn 1.7.2，imbalanced-learn 0.14.0，shap 0.54.1，pandas 2.3.3。
- 统一口径：`SEED=42`，标签 `default`（1=违约），`COST_FALSE_NEGATIVE=10`，`COST_FALSE_POSITIVE=1`，阈值网格 0.05–0.95。

## 数据获取（真实下载）

| 命令 | 实测耗时 | 产物 |
|---|---|---|
| `python scripts/download_data.py` | 4.14 s（其中下载 2.44 s） | 官方 zip 5,539,494 B；`data/raw/credit_default.csv` 30,000×24，2,698,292 B |

- zip sha256：`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`
- csv sha256：`e28803eec99215182faffbced63edd940d5034ed240443084ae72d254f147f89`
- 数据源：UCI Machine Learning Repository, dataset id=350，官方 `default of credit card clients.xls`；未使用匿名列名的 `ucimlrepo` 版本。

## 官方全量实验（2026-11-01 21:58–22:03）

命令：

```bash
python3 scripts/run_experiments.py
```

实测总耗时 **505.5 s**，共 **81 次运行**。分阶段耗时（取自运行日志）：

| 阶段 | 起止（累计秒） | 耗时 | 内容 |
|---|---|---|---|
| 数据加载与质量诊断 | 0.0–2.6 | 2.6 s | t03/t04、f01–f04 |
| 主结果 | 2.6–158.2 | 155.6 s | 4 模型 × 2–3 不平衡策略 × 5 种子，t05/t06、f05 |
| 超参搜索 | 158.2–407.0 | 248.8 s | LightGBM/XGBoost 各 9 组 × 5 折 CV，t07 |
| 概率校准 | 407.0–451.3 | 44.3 s | 2 模型 × 3 校准 × 3 种子，t08、f06 |
| 代价敏感阈值 | 451.3–475.2 | 23.9 s | FN:FP=5:1/10:1/20:1，t09、f07 |
| 消融 | 475.2–494.9 | 19.7 s | 6 个变体，t10、f08 |
| SHAP 与混淆矩阵 | 495.1–505.5 | 10.4 s | t11、f09–f12 |

数据口径：原始 30,000 行，整行重复 35 条（0.12%）去重后 **29,965 行**；分层 70/15/15 = **20,975 / 4,495 / 4,495**（正类 4,641 / 995 / 994，正类率 22.13% / 22.14% / 22.11%）；特征 **43 个**（23 原始 + 20 衍生）。划分索引 `data/processed/split_index.json`，所有实验共用。

种子策略：主结果表 10 个“模型 × 策略”组合 × 5 个种子（42、2024、7、123、888）；校准 3 个种子；消融与代价敏感单种子。该取舍符合“5 种子只用于主结果表、消融单种子”的预算约束。

## 补充实验

| 命令 | 实测耗时 | 产物 |
|---|---|---|
| `python3 scripts/extra_calibration_imbalance.py` | 30 次运行，3 种不平衡策略 × 3 种校准 × 3 种子 | t12、`metrics/calibration_imbalance.json` |
| `python3 scripts/extra_error_analysis.py` | 单次训练 + 分箱统计 | t13、t14、f13 |

补充实验结论（均为实测）：

- **类别加权破坏概率校准**：`class_weight` 使 ECE 由 0.0216（无加权）升至 0.1374、Brier 由
  0.1352 升至 0.1614；重新校准后 ECE 回到 0.0154（sigmoid 为 0.0145）。
- **校准恢复了加权的业务价值**：`class_weight` 未校准时成本节省率仅 31.56%，校准后为 53.29%
  （isotonic）/ 55.27%（sigmoid），与不加权的 51%–53% 基本持平。
- **误差结构**：高置信误判（p≥0.8 却未违约）仅占测试集 0.22%，高置信漏判（p≤0.2 却违约）
  占 8.3%，与 FN:FP=10:1 的代价设定一致。

## 产物清单

- `results/tables/t03–t14_*.csv|md`：报告表格数据源。
- `results/metrics/all_runs.json`：81 次运行的完整指标；`data_quality.json`：数据质量计数；
  `calibration_imbalance.json`：补充实验逐次记录。
- `results/figures/f01–f13_*.png`：300 dpi，中文标签。
- 绘图所需的中文字体由脚本首次运行时从系统 Noto CJK 字体集自动提取到 `assets/fonts/`（不入库）。
