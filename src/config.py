# -*- coding: utf-8 -*-
"""全局配置：路径、随机种子、评估与绘图参数。

约定
----
1. 所有路径常量集中在此文件，脚本内禁止硬编码绝对路径。
2. 所有随机过程统一读取 ``SEED``，保证实验可复现。
3. 评估口径（主指标、阈值搜索范围）也在此定义，避免各处口径不一致。
"""

from __future__ import annotations

from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
METRIC_DIR = RESULTS_DIR / "metrics"
MODEL_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR, METRIC_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 数据与任务 ----------
DATASET_NAME = "credit_default"   # data/raw/<DATASET_NAME>.csv
TARGET_COL = "default"             # 标签列：是否违约（1 = 违约）
POSITIVE_LABEL = 1

# ---------- 实验设置 ----------
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
N_FOLDS = 5
N_JOBS = 4

# ---------- 不平衡处理与代价敏感 ----------
# 误判代价比：漏判（违约判为正常）代价 / 误杀（正常判为违约）代价
COST_FALSE_NEGATIVE = 10.0
COST_FALSE_POSITIVE = 1.0
# 概率阈值搜索范围（代价敏感决策用，不直接取 0.5）
THRESHOLD_GRID = [i / 100 for i in range(5, 96, 1)]

# ---------- 输出设置 ----------
FIG_DPI = 300
FIG_FORMAT = "png"
TABLE_FORMAT = "csv"

# ---------- 绘图统一风格 ----------
PLOT_STYLE = {
    "font.size": 10.5,
    "axes.unicode_minus": False,
    "figure.autolayout": True,
    "savefig.bbox": "tight",
    "savefig.dpi": FIG_DPI,
}
