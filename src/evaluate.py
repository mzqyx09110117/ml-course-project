# -*- coding: utf-8 -*-
"""评估指标、概率校准质量与代价敏感决策。

指标口径（报告中需与本节完全一致）
--------------------------------
- AUC：ROC 曲线下面积，衡量排序能力，对类别不平衡不敏感；
- AP：PR 曲线下面积（average precision），不平衡场景下比 AUC 更敏感；
- F1：给定决策阈值下的调和平均；
- KS：max(TPR - FPR)，风控行业常用区分度指标；
- Brier：概率预测的均方误差，同时惩罚"排序错"和"概率偏"；
- ECE：期望校准误差（等宽 10 箱），衡量预测概率与真实频率的偏离；
- 期望误判成本：``CFN * FN + CFP * FP``，业务上真正关心的量；
- 成本节省率：相对"固定 0.5 阈值"的期望成本下降比例。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, THRESHOLD_GRID


def expected_cost(y_true, y_prob, threshold, c_fp=COST_FALSE_POSITIVE, c_fn=COST_FALSE_NEGATIVE):
    """期望误判成本：漏判（FN）与误杀（FP）分别按各自代价计费。"""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(c_fn * fn + c_fp * fp), (int(tn), int(fp), int(fn), int(tp))


def best_threshold(y_true, y_prob, c_fp=COST_FALSE_POSITIVE, c_fn=COST_FALSE_NEGATIVE,
                   grid=None):
    """在验证集上按期望误判成本最小化搜索阈值。

    解析解为 ``p* = CFP / (CFP + CFN)``（概率校准良好时最优），这里同时给出网格搜索结果，
    用于对比"解析阈值"与"数据驱动阈值"，并画出成本曲线。
    """
    grid = THRESHOLD_GRID if grid is None else grid
    costs = [expected_cost(y_true, y_prob, t, c_fp, c_fn)[0] for t in grid]
    i = int(np.argmin(costs))
    analytic = c_fp / (c_fp + c_fn)
    return {
        "threshold_grid": float(grid[i]),
        "threshold_analytic": float(analytic),
        "cost_curve": {"threshold": list(map(float, grid)), "cost": [float(c) for c in costs]},
    }


def ece_score(y_true, y_prob, n_bins: int = 10):
    """期望校准误差（等宽分箱）与可靠性曲线数据。"""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    rows = []
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append({"bin": b, "mean_pred": float((bins[b] + bins[b + 1]) / 2),
                         "frac_pos": np.nan, "count": 0})
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        ece += abs(acc - conf) * m.sum() / n
        rows.append({"bin": b, "mean_pred": conf, "frac_pos": acc, "count": int(m.sum())})
    return float(ece), rows


def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, y_prob, threshold: float,
                    c_fp=COST_FALSE_POSITIVE, c_fn=COST_FALSE_NEGATIVE) -> dict:
    """一次性算出报告所需的全部指标（含混淆矩阵与代价）。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    cost, (tn, fp, fn, tp) = expected_cost(y_true, y_prob, threshold, c_fp, c_fn)
    ece, _ = ece_score(y_true, y_prob)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": ece,
        "threshold": float(threshold),
        "expected_cost": cost,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
    }


def curve_data(y_true, y_prob):
    """ROC 与 PR 曲线数据（绘图用）。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "precision": prec.tolist(), "recall": rec.tolist()}