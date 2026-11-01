#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据获取脚本：下载原始数据到 ``data/raw/``。

数据本身不入版本库（见 ``.gitignore``），评审者运行本脚本即可复现。
依赖：``ucimlrepo``（UCI 官方 Python 客户端）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# TODO(数据与实验)：确认数据集编号与名称后填写
DATASET_ID = 350          # UCI 数据集编号（待确认）
DATASET_NAME = "credit_default"   # 输出文件名 data/raw/<name>.csv


def fetch_uci(dataset_id: int, out_name: str) -> Path:
    """按 UCI 数据集编号下载，合并特征与标签后落盘为 CSV。"""
    from ucimlrepo import fetch_ucirepo

    print(f"[下载] UCI 数据集 id={dataset_id}")
    payload = fetch_ucirepo(id=dataset_id)
    df = payload.data.features.copy()
    if payload.data.targets is not None:
        for col in payload.data.targets.columns:
            df[col] = payload.data.targets[col]

    out = RAW_DIR / f"{out_name}.csv"
    df.to_csv(out, index=False)
    print(f"[完成] {out}  shape={df.shape}")
    return out


def verify(path: Path) -> None:
    """下载后校验：行数、列数、标签列是否存在。"""
    import pandas as pd

    df = pd.read_csv(path)
    print(f"[校验] shape={df.shape}  columns={len(df.columns)}")
    # TODO(数据与实验)：补充期望形状断言与标签列检查
    if df.empty:
        raise ValueError("数据为空，检查下载源")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载原始数据集")
    ap.add_argument("--id", type=int, default=DATASET_ID, help="UCI 数据集编号")
    ap.add_argument("--name", default=DATASET_NAME, help="输出文件名（不含扩展名）")
    args = ap.parse_args()

    path = fetch_uci(args.id, args.name)
    verify(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
