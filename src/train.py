# -*- coding: utf-8 -*-
"""训练流程：拟合、概率校准、代价敏感阈值决策、交叉验证。

训练/校准/阈值三段严格分离，防止信息泄漏
------------------------------------
1. 分类器只在 **训练集** 上拟合；
2. 概率校准器只在 **验证集** 上拟合（模型参数冻结）；
3. 决策阈值只在 **验证集** 上按期望误判成本搜出；
4. 测试集只在最后评估时使用一次。
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from .config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    N_FOLDS,
    SEED,
)
from .evaluate import best_threshold, compute_metrics
from .models import balanced_pipeline, fit_pipeline


def calibrate(estimator, X_val, y_val, method: str = "none"):
    """概率校准：none / sigmoid(Platt) / isotonic。"""
    if method == "none":
        return estimator
    try:  # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator
        cc = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except Exception:  # 旧版本
        cc = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    cc.fit(X_val, y_val)
    return cc


def positive_scores(model, X) -> np.ndarray:
    """统一取出正类（违约）概率。"""
    return model.predict_proba(X)[:, 1]


def train_one(model_name: str, X_tr, y_tr, X_va, y_va, X_te, y_te,
              strategy: str = "none", calibration: str = "none",
              seed: int = SEED, use_cost_threshold: bool = True,
              c_fp: float = COST_FALSE_POSITIVE, c_fn: float = COST_FALSE_NEGATIVE,
              **model_kwargs):
    """跑通"训练 → 校准 → 阈值决策 → 测试评估"一条链路，返回指标字典。"""
    pipe = balanced_pipeline(model_name, strategy=strategy, seed=seed, **model_kwargs)
    pipe = fit_pipeline(pipe, X_tr, y_tr, X_tr.columns, strategy=strategy)

    model = calibrate(pipe, X_va, y_va, calibration)
    p_va_c = positive_scores(model, X_va)

    thr_info = best_threshold(y_va, p_va_c, c_fp, c_fn)
    threshold = thr_info["threshold_grid"] if use_cost_threshold else 0.5

    p_te = positive_scores(model, X_te)
    res = compute_metrics(y_te, p_te, threshold, c_fp, c_fn)
    base = compute_metrics(y_te, p_te, 0.5, c_fp, c_fn)
    res.update({
        "model": model_name,
        "strategy": strategy,
        "calibration": calibration,
        "seed": seed,
        "cost_ratio": f"{int(c_fn)}:{int(c_fp)}",
        "use_cost_threshold": bool(use_cost_threshold),
        "cost_at_05": base["expected_cost"],
        "cost_saving": (float(1 - res["expected_cost"] / base["expected_cost"])
                        if base["expected_cost"] > 0 else 0.0),
        "val_threshold": threshold,
        "threshold_analytic": thr_info["threshold_analytic"],
    })
    return res, model, p_te, thr_info


def cross_validate(model_name: str, X, y, strategy: str = "none", folds: int = N_FOLDS,
                   seed: int = SEED, **model_kwargs) -> dict:
    """5 折分层交叉验证（仅用于模型选择，不参与最终测试集评估）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs, aps = [], []
    for tr, va in skf.split(X, y):
        pipe = balanced_pipeline(model_name, strategy=strategy, seed=seed, **model_kwargs)
        pipe = fit_pipeline(pipe, X.iloc[tr], y[tr], X.columns, strategy=strategy)
        p = positive_scores(pipe, X.iloc[va])
        aucs.append(roc_auc_score(y[va], p))
        aps.append(average_precision_score(y[va], p))
    return {"model": model_name, "strategy": strategy, "folds": folds,
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps)),
            "auc_folds": [float(a) for a in aucs], "ap_folds": [float(a) for a in aps]}