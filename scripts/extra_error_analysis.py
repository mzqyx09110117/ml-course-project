#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充实验：误差分析（高置信误判个案）。

回答两个问题：
1. 模型"非常自信却判错"的样本有多少？
2. 这些样本在关键特征上有什么共性（是数据问题还是模型能力上限）？
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data as D
from src.config import FIG_DIR, FIG_FORMAT, PLOT_STYLE, SEED, TABLE_DIR
from src.train import positive_scores, train_one

plt.rcParams.update(PLOT_STYLE)
_font = ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
if _font.exists():
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(str(_font))
    plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=str(_font)).get_name(),
                                       "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CONF_HIGH = 0.8
CONF_LOW = 0.2


def main() -> int:
    feat, y = D.build_dataset(feature_set="full")
    tr, va, te = D.make_split(y, seed=SEED)
    X_tr, X_va, X_te = feat.iloc[tr], feat.iloc[va], feat.iloc[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    res, _, p_te, _ = train_one(
        "lgbm", X_tr, y_tr, X_va, y_va, X_te, y_te,
        strategy="none", calibration="isotonic", seed=SEED)

    test = X_te.copy()
    test["y_true"] = y_te
    test["p_hat"] = p_te
    test["kind"] = np.where(
        (test.p_hat >= CONF_HIGH) & (test.y_true == 0), "高置信误判(FP)",
        np.where((test.p_hat <= CONF_LOW) & (test.y_true == 1), "高置信漏判(FN)", "正常"))

    summary = (test.groupby("kind")
                   .agg(样本数=("p_hat", "size"), 平均预测概率=("p_hat", "mean"),
                        平均逾期次数=("LATE_CNT1", "mean"),
                        平均信用利用率=("UTIL_MAX", "mean"),
                        平均授信额度=("LIMIT_BAL", "mean"),
                        平均年龄=("AGE", "mean")).reset_index()
                   .rename(columns={"kind": "样本类型"}))
    summary["占比"] = summary["样本数"] / len(test)
    summary.to_csv(TABLE_DIR / "t13_error_analysis.csv", index=False, encoding="utf-8-sig")
    (TABLE_DIR / "t13_error_analysis.md").write_text(
        summary.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(summary.to_string(index=False))

    # 典型个案（按与决策边界距离排序，取最"自信"的 5 例）
    fp = test[test.kind.str.contains("FP")].nlargest(5, "p_hat")
    fn = test[test.kind.str.contains("FN")].nsmallest(5, "p_hat")
    cases = pd.concat([fp, fn])
    cols = ["LATE_CNT1", "PAY_0", "UTIL_MAX", "PAY_RATIO_MEAN", "LIMIT_BAL", "AGE",
            "p_hat", "y_true"]
    cases[cols].to_csv(TABLE_DIR / "t14_error_cases.csv", index=False, encoding="utf-8-sig")
    (TABLE_DIR / "t14_error_cases.md").write_text(
        cases[cols].to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for kind, color, marker in [("正常", "#9aa4b8", "o"), ("高置信误判(FP)", "#f2705b", "^"),
                                ("高置信漏判(FN)", "#5b8cff", "s")]:
        d = test[test.kind == kind]
        ax.scatter(d["LATE_CNT1"] + np.random.RandomState(0).normal(0, .08, len(d)),
                   d["p_hat"], s=6 if kind == "正常" else 26,
                   alpha=.25 if kind == "正常" else .85,
                   c=color, marker=marker, label=kind)
    ax.axhline(CONF_HIGH, ls="--", lw=.7, c="k")
    ax.axhline(CONF_LOW, ls="--", lw=.7, c="k")
    ax.set_xlabel("逾期次数 LATE_CNT1（含抖动）"); ax.set_ylabel("预测违约概率")
    ax.set_title("高置信误判样本分布"); ax.legend(fontsize=8)

    ax = axes[1]
    err_rate = (test.assign(bucket=pd.cut(test.p_hat, np.linspace(0, 1, 11)))
                    .groupby("bucket", observed=True)
                    .agg(样本数=("y_true", "size"), 实际违约率=("y_true", "mean"),
                         平均预测=("p_hat", "mean")).reset_index())
    ax.plot(err_rate["平均预测"], err_rate["实际违约率"], "o-", label="实际违约率")
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="理想校准线")
    for _, r in err_rate.iterrows():
        ax.annotate(f"{int(r['样本数'])}", (r["平均预测"], r["实际违约率"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("平均预测概率"); ax.set_ylabel("实际违约率")
    ax.set_title("测试集分箱校准（标注为样本数）"); ax.legend(fontsize=8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"f13_error_cases.{FIG_FORMAT}", dpi=300, bbox_inches="tight")
    print("saved f13_error_cases")
    print("高置信误判占比: %.4f" % (len(fp) / len(test)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())