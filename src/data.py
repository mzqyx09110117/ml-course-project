# -*- coding: utf-8 -*-
"""数据读取、清洗、特征工程与划分。

实现说明
--------
1. 官方 xls 的列名即真实字段名；部分获取方式（例如 ``ucimlrepo`` 接口）返回的是匿名列名
   ``X1..X23``，因此这里保留一份"按官方字段顺序"的映射做兼容，避免后续图表出现 ``X17`` 这类列名。
2. 清洗规则与"未文档化编码"的处置都在 :func:`clean` 中显式实现，并在报告中逐条说明。
3. 特征工程分两组：账单序列特征（均值/波动/趋势）与还款行为特征（利用率/还款比/逾期计数）。
   两组特征都可单独开关，用于消融实验与"特征可得性削减"实验。
4. 划分固定为分层 70/15/15，索引落盘，保证所有实验共用同一划分。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)

# ---------- 字段映射 ----------
# 官方字段顺序（UCI id=350），ucimlrepo 返回 X1..X23 时按下表还原
COLUMN_MAP = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
    "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
    "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
    "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
    "Y": TARGET_COL,
}

# 类别型字段（树模型直接用整数编码；线性模型做 one-hot）
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
# 还款状态码（PAY_*）：-2 无消费, -1 按期还款, 0 最低还款, 1..9 逾期月数
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]      # BILL_AMT1 为最近一期
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# 人口统计字段（"冷启动/特征可得性削减"实验里只保留这些）
DEMO_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]


def load_raw(name: str = "credit_default") -> pd.DataFrame:
    """读取 ``data/raw/<name>.csv`` 并还原字段名。"""
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少原始数据 {path}，请先运行: python scripts/download_data.py"
        )
    df = pd.read_csv(path)
    rename = {c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP}
    df = df.rename(columns=rename)
    if TARGET_COL not in df.columns:
        df = df.rename(columns={df.columns[-1]: TARGET_COL})
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗：重复行、未文档化类别编码、类型规范。

    - 完全重复记录：去重；
    - EDUCATION 取值 0/5/6 官方文档未定义 → 归入 4（其他）；
    - MARRIAGE 取值 0 未定义 → 归入 3（其他）；
    - 缺失值：本数据集无缺失，仍做一次显式检查以防脏数据。
    """
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    for col in df.columns:
        if col != TARGET_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def _slope(mat: np.ndarray) -> np.ndarray:
    """按时间顺序（最近一期末）对 6 期序列拟合一次趋势斜率。"""
    x = np.arange(mat.shape[1], dtype=float)
    x = x - x.mean()
    denom = (x ** 2).sum()
    return (mat - mat.mean(axis=1, keepdims=True)) @ x / denom


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程：账单序列特征 + 还款行为特征。"""
    df = df.copy()
    bill = df[BILL_COLS].to_numpy(dtype=float)
    pay = df[PAY_AMT_COLS].to_numpy(dtype=float)
    limit = df["LIMIT_BAL"].to_numpy(dtype=float)

    bill_chrono = bill[:, ::-1]
    df["BILL_AVG"] = bill.mean(axis=1)
    df["BILL_STD"] = bill.std(axis=1)
    df["BILL_MAX"] = bill.max(axis=1)
    df["BILL_MIN"] = bill.min(axis=1)
    df["BILL_SLOPE"] = _slope(bill_chrono)
    df["BILL_RANGE"] = df["BILL_MAX"] - df["BILL_MIN"]

    df["UTIL_CUR"] = bill[:, 0] / (limit + 1.0)
    df["UTIL_AVG"] = bill.mean(axis=1) / (limit + 1.0)
    df["UTIL_MAX"] = bill.max(axis=1) / (limit + 1.0)

    ratio = np.zeros((bill.shape[0], 5), dtype=float)
    for t in range(5):
        ratio[:, t] = pay[:, t] / (np.abs(bill[:, t]) + 1.0)
    df["PAY_RATIO_MEAN"] = ratio.mean(axis=1)
    df["PAY_RATIO_MIN"] = ratio.min(axis=1)

    df["PAY_AMT_AVG"] = pay.mean(axis=1)
    df["PAY_AMT_SUM"] = pay.sum(axis=1)
    df["PAY_TO_BILL"] = pay.sum(axis=1) / (np.abs(bill).sum(axis=1) + 1.0)
    df["ZERO_PAY_CNT"] = (pay == 0).sum(axis=1)

    pay_status = df[PAY_STATUS].to_numpy(dtype=float)
    df["LATE_CNT1"] = (pay_status >= 1).sum(axis=1)
    df["LATE_CNT2"] = (pay_status >= 2).sum(axis=1)
    df["MAX_DELAY"] = pay_status.max(axis=1)
    df["DELAY_SUM"] = pay_status.clip(min=0).sum(axis=1)
    df["DELAY_TREND"] = _slope(pay_status[:, ::-1])

    for col in ["UTIL_CUR", "UTIL_AVG", "UTIL_MAX", "PAY_RATIO_MEAN", "PAY_RATIO_MIN",
                "PAY_TO_BILL", "BILL_SLOPE", "DELAY_TREND"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].clip(df[col].quantile(0.001), df[col].quantile(0.999))
        df[col] = df[col].fillna(0.0)
    return df


def build_dataset(name: str = "credit_default", feature_set: str = "full"):
    """原始数据 → 清洗 → 特征工程 → (X, y)。

    feature_set : full 全部派生特征 / raw 仅原始字段（消融）/ demo 仅人口统计字段（冷启动）。
    """
    df = clean(load_raw(name))
    if feature_set == "raw":
        feat = df.drop(columns=[TARGET_COL])
    elif feature_set == "demo":
        feat = df[DEMO_COLS].copy()
    elif feature_set == "full":
        feat = add_features(df).drop(columns=[TARGET_COL])
    else:
        raise ValueError(f"未知 feature_set: {feature_set}")
    y = df[TARGET_COL].to_numpy().astype(int)
    return feat, y


def make_split(y: np.ndarray, seed: int = SEED):
    """分层 70/15/15 划分，索引落盘到 ``data/processed/split_index.json``。"""
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=TEST_SIZE + VAL_SIZE, stratify=y, random_state=seed)
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    idx_va, idx_te = train_test_split(
        idx_tmp, test_size=1 - val_ratio, stratify=y[idx_tmp], random_state=seed)
    split = {"train": idx_tr.tolist(), "val": idx_va.tolist(), "test": idx_te.tolist()}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "split_index.json").write_text(json.dumps(split), encoding="utf-8")
    return idx_tr, idx_va, idx_te


def data_quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """输出可写进报告表的数据质量统计（全部为真实计数）。"""
    return {
        "n_rows_raw": int(len(df_raw)),
        "n_cols": int(df_raw.shape[1]),
        "n_dup_rows": int(len(df_raw) - len(df_raw.drop_duplicates())),
        "n_rows_clean": int(len(df_clean)),
        "pos_rate": float(df_clean[TARGET_COL].mean()),
        "n_missing": int(df_clean.isna().sum().sum()),
        "n_edu_undoc": int((df_raw["EDUCATION"].isin([0, 5, 6])).sum()),
        "n_marriage_undoc": int((df_raw["MARRIAGE"] == 0).sum()),
        "n_neg_bill": int((df_raw[BILL_COLS] < 0).sum().sum()),
        "neg_bill_rate": float((df_raw[BILL_COLS] < 0).to_numpy().mean()),
        "n_allzero_pay": int((df_raw[PAY_AMT_COLS].sum(axis=1) == 0).sum()),
        "limit_min": float(df_clean["LIMIT_BAL"].min()),
        "limit_max": float(df_clean["LIMIT_BAL"].max()),
        "age_min": int(df_clean["AGE"].min()),
        "age_max": int(df_clean["AGE"].max()),
    }