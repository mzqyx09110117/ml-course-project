#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据获取脚本：下载 UCI Default of Credit Card Clients 并转为 CSV。

数据本身不入版本库（见 ``.gitignore``），评审者运行本脚本即可复现。
依赖：``pandas``、``xlrd``（读取官方 .xls）。
官方来源：UCI Machine Learning Repository, dataset id=350。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
DATASET_NAME = "credit_default"
XLS_NAME = "default of credit card clients.xls"
TARGET_RAW = "default payment next month"
TARGET_COL = "default"
EXPECTED_SHAPE = (30000, 24)   # 23 个特征 + 1 个标签（不含 ID）


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out: Path) -> Path:
    if out.exists():
        print(f"[跳过] 已存在 {out}，如需重新下载请先删除")
        return out
    print(f"[下载] {url}")
    t0 = time.time()
    urllib.request.urlretrieve(url, out)
    print(f"[完成] {out}  {out.stat().st_size} bytes  {time.time() - t0:.2f} s")
    return out


def convert(zip_path: Path) -> Path:
    """解压官方 xls，去掉无意义主键 ID，统一标签列名为 default。"""
    import pandas as pd

    out = RAW_DIR / f"{DATASET_NAME}.csv"
    with tempfile.TemporaryDirectory(dir=RAW_DIR) as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(XLS_NAME, tmp)
        xls = Path(tmp) / XLS_NAME
        df = pd.read_excel(xls, header=1)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df.rename(columns={TARGET_RAW: TARGET_COL})
    df.to_csv(out, index=False)
    print(f"[完成] {out}  shape={df.shape}")
    return out


def verify(path: Path) -> None:
    """下载后校验：行数、列数、标签分布、缺失与重复。"""
    import pandas as pd

    df = pd.read_csv(path)
    print(f"[校验] shape={df.shape}  columns={len(df.columns)}")
    assert df.shape == EXPECTED_SHAPE, f"形状异常：{df.shape} != {EXPECTED_SHAPE}"
    assert TARGET_COL in df.columns, f"缺少标签列 {TARGET_COL}"
    assert df[TARGET_COL].isin([0, 1]).all(), "标签取值应为 0/1"
    assert df.isna().sum().sum() == 0, "存在缺失值"
    print(f"[校验] 标签分布={df[TARGET_COL].value_counts().to_dict()}  "
          f"重复行={int(df.duplicated().sum())}")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载信用违约数据集")
    ap.add_argument("--url", default=DATASET_URL, help="官方 zip 地址")
    ap.add_argument("--name", default=DATASET_NAME, help="输出文件名（不含扩展名）")
    args = ap.parse_args()

    zip_path = RAW_DIR / "default_credit.zip"
    t0 = time.time()
    download(args.url, zip_path)
    print(f"[校验] zip sha256={sha256_file(zip_path)}")
    out = convert(zip_path)
    print(f"[校验] csv sha256={sha256_file(out)}  size={out.stat().st_size} bytes")
    verify(out)
    print(f"[总耗时] {time.time() - t0:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
