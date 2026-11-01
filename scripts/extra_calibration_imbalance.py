#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充实验：校准在不同不平衡策略下的作用。

动机：主结果表显示 ``class_weight`` 会显著抬高 ECE（概率校准变差），
本节定量回答"校准能否把加权模型破坏的概率质量修回来"，并把结论写进报告第 5 章。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D
from src.config import METRIC_DIR, SEED, TABLE_DIR
from src.models import MODEL_CN
from src.train import train_one

OUT = TABLE_DIR / "t12_calibration_imbalance"
FIG_COLS = ["strategy", "calibration", "auc", "ap", "brier", "ece",
            "threshold", "expected_cost", "cost_saving", "n"]


def main() -> int:
    feat, y = D.build_dataset(feature_set="full")
    tr, va, te = D.make_split(y, seed=SEED)
    X_tr, X_va, X_te = feat.iloc[tr], feat.iloc[va], feat.iloc[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    rows = []
    for strategy in ["none", "class_weight", "smote"]:
        for cal in ["none", "sigmoid", "isotonic"]:
            for sd in [SEED, 2024, 7]:
                res, _, _, _ = train_one(
                    "lgbm", X_tr, y_tr, X_va, y_va, X_te, y_te,
                    strategy=strategy, calibration=cal, seed=sd)
                rows.append(res)

    df = pd.DataFrame(rows)
    agg = (df.groupby(["strategy", "calibration"])
             .agg(auc=("auc", "mean"), ap=("ap", "mean"),
                  brier=("brier", "mean"), ece=("ece", "mean"),
                  threshold=("threshold", "mean"),
                  expected_cost=("expected_cost", "mean"),
                  cost_saving=("cost_saving", "mean"), n=("auc", "size"))
             .reset_index())
    agg["模型"] = "LightGBM"
    agg[FIG_COLS].to_csv(f"{OUT}.csv", index=False, encoding="utf-8-sig")
    (TABLE_DIR / f"{OUT.name}.md").write_text(
        agg[FIG_COLS].to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(agg[FIG_COLS].to_string(index=False))
    (METRIC_DIR / "calibration_imbalance.json").write_text(
        df.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())