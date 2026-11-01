#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高置信错误个案分析：挑出模型最有把握却判错的样本，给出 SHAP 归因。

选择规则（测试集，完整方法 = LightGBM + Isotonic 校准）：
- 高置信误杀（FP）：真实未违约，校准概率 >= 0.90，取概率最高者；
- 高置信漏判（FN）：真实违约，校准概率 <= 0.05，取概率最低者。

产物：
- results/tables/t12_error_cases.csv / .md
- results/figures/f13_error_cases.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D
from src.config import FIG_DIR, SEED, TABLE_DIR
from src.evaluate import best_threshold, compute_metrics
from src.models import balanced_pipeline, fit_pipeline
from src.train import calibrate, positive_scores
from src.utils import apply_plot_style, setup_cjk_font

SHOW_COLS = [
    "PAY_0", "LATE_CNT1", "LATE_CNT2", "UTIL_MAX", "PAY_RATIO_MEAN",
    "BILL_AVG", "PAY_AMT_AVG", "LIMIT_BAL", "AGE",
]


def fmt_contrib(feature: str, value: float, shap_value: float) -> str:
    return f"{feature}={value:g} (SHAP {shap_value:+.3f})"


def main() -> int:
    feat, y = D.build_dataset(feature_set="full")
    idx_tr, idx_va, idx_te = D.make_split(y, seed=SEED)
    X_tr, X_va, X_te = feat.iloc[idx_tr], feat.iloc[idx_va], feat.iloc[idx_te]
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    pipe = fit_pipeline(
        balanced_pipeline("lgbm", strategy="none", seed=SEED),
        X_tr, y_tr, feat.columns,
    )
    model = calibrate(pipe, X_va, y_va, "isotonic")
    p_va = positive_scores(model, X_va)
    p_te = positive_scores(model, X_te)
    threshold = best_threshold(y_va, p_va, 1.0, 10.0)["threshold_grid"]
    metrics = compute_metrics(y_te, p_te, threshold)

    fp_candidates = np.where((y_te == 0) & (p_te >= 0.90))[0]
    fn_candidates = np.where((y_te == 1) & (p_te <= 0.05))[0]
    if len(fp_candidates) == 0 or len(fn_candidates) == 0:
        raise RuntimeError("测试集中没有满足阈值的高置信错误样本")

    i_fp = int(fp_candidates[np.argmax(p_te[fp_candidates])])
    i_fn = int(fn_candidates[np.argmin(p_te[fn_candidates])])

    import shap
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = explainer.shap_values(X_te.iloc[[i_fp, i_fn]])
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)

    rows = []
    for case, pos, label_cn in [("FP", i_fp, "真实正常→预测违约"), ("FN", i_fn, "真实违约→预测正常")]:
        row = {
            "case": case,
            "description": label_cn,
            "test_row": pos,
            "true_label": int(y_te[pos]),
            "p_calibrated": float(p_te[pos]),
            "p_raw": float(positive_scores(pipe, X_te.iloc[[pos]])[0]),
            "decision_threshold": float(threshold),
        }
        for col in SHOW_COLS:
            row[col] = float(X_te.iloc[pos][col])
        local = sv[0] if pos == i_fp else sv[1]
        contrib = sorted(
            zip(X_te.columns, X_te.iloc[pos].to_numpy(), local),
            key=lambda t: abs(t[2]), reverse=True,
        )[:6]
        row["top_shap"] = "; ".join(fmt_contrib(f, v, s) for f, v, s in contrib)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "t12_error_cases.csv", index=False, encoding="utf-8-sig")
    (TABLE_DIR / "t12_error_cases.md").write_text(
        df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8"
    )
    print(df[["case", "test_row", "true_label", "p_calibrated", "p_raw", "top_shap"]].to_string(index=False))

    apply_plot_style()
    setup_cjk_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, pos, title in [(axes[0], i_fp, "高置信误杀（FP）"), (axes[1], i_fn, "高置信漏判（FN）")]:
        local = sv[0] if pos == i_fp else sv[1]
        order = np.argsort(-np.abs(local))[:8][::-1]
        colors = ["#f2705b" if local[j] > 0 else "#5b8cff" for j in order]
        ax.barh(range(len(order)), local[order], color=colors)
        ax.set_yticks(range(len(order)), [X_te.columns[j] for j in order], fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("SHAP 值（正=推高违约概率）")
        ax.set_title(f"{title}：真值={int(y_te[pos])}，p={p_te[pos]:.3f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f13_error_cases.png", dpi=300, bbox_inches="tight")
    print(f"[图] {FIG_DIR / 'f13_error_cases.png'}")
    print(f"[指标] 完整方法阈值={threshold:.2f} 测试集 AUC={metrics['auc']:.4f} 成本={metrics['expected_cost']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
