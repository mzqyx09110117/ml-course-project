# -*- coding: utf-8 -*-
"""数据读取、清洗与划分。

模块接口已固定（见下），具体实现由实验部分填充，避免训练/评估脚本反复改动。

实现要求
--------
- 只从 ``config.RAW_DIR`` 读取，禁止硬编码路径；
- 清洗与划分必须是确定性的：固定 ``config.SEED``；
- 划分索引落盘（``data/processed/split_index.json``），保证每次运行使用同一划分。
"""

from __future__ import annotations

import pandas as pd

from .config import RAW_DIR, SEED, TARGET_COL


def load_raw(name: str) -> pd.DataFrame:
    """读取 ``data/raw/<name>.csv``。

    Raises
    ------
    FileNotFoundError
        数据文件缺失时提示先运行 ``scripts/download_data.py``。
    """
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少原始数据 {path}，请先运行: python scripts/download_data.py"
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗：去重、异常值处理、类型规范化。TODO(实验部分)"""
    raise NotImplementedError("clean() 待实现：去重 / 缺失值 / 异常值 / 类型规范化")


def build_dataset(name: str = "credit_default", target: str = TARGET_COL):
    """原始数据 → 清洗 → 特征矩阵 X 与标签 y。TODO(实验部分)"""
    raise NotImplementedError("build_dataset() 待实现")


def make_split(df: pd.DataFrame, seed: int = SEED):
    """按 ``config.TEST_SIZE`` / ``VAL_SIZE`` 做分层划分，并落盘索引。TODO(实验部分)"""
    raise NotImplementedError("make_split() 待实现：分层划分 + 索引落盘")
