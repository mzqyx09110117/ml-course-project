# -*- coding: utf-8 -*-
"""模型工厂与不平衡处理管线。

设计要点
--------
1. 所有模型统一暴露 ``fit / predict_proba`` 接口，便于后续统一做概率校准与阈值决策。
2. 类别不平衡策略（``class_weight`` / ``smote`` / ``undersample``）通过 :func:`balanced_pipeline`
   统一包装；采样器只作用于训练折，从结构上杜绝"先过采样再划分"造成的数据泄漏。
3. 线性模型需要 one-hot + 标准化，树模型直接用整数编码，差异由 :func:`make_preprocessor` 处理。
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import N_JOBS, SEED

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:  # pragma: no cover
    SMOTE = None
    ImbPipeline = None
    RandomUnderSampler = None

CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def make_preprocessor(feature_names, scale: bool = True) -> ColumnTransformer:
    """类别列 one-hot、数值列标准化（仅线性模型需要）。"""
    cat_cols = [c for c in CAT_COLS if c in feature_names]
    num_cols = [c for c in feature_names if c not in cat_cols]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler() if scale else "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(name: str, seed: int = SEED, **kwargs):
    """按名称构造未训练的估计器（线性模型的预处理在 :func:`balanced_pipeline` 中拼接）。"""
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000, solver="lbfgs", n_jobs=N_JOBS, random_state=seed, **kwargs)
    if name == "rf":
        params = dict(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                      n_jobs=N_JOBS, random_state=seed)
        params.update(kwargs)
        return RandomForestClassifier(**params)
    if name == "lgbm":
        if lgb is None:
            raise ImportError("需要 lightgbm")
        params = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                      subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                      min_child_samples=40, reg_lambda=1.0,
                      n_jobs=N_JOBS, random_state=seed, verbose=-1)
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)
    if name == "xgb":
        if xgb is None:
            raise ImportError("需要 xgboost")
        params = dict(n_estimators=500, learning_rate=0.05, max_depth=5,
                      subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                      reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
                      n_jobs=N_JOBS, random_state=seed)
        params.update(kwargs)
        return xgb.XGBClassifier(**params)
    raise ValueError(f"未知模型: {name}")


MODEL_ZOO = ["logreg", "rf", "lgbm", "xgb"]
MODEL_CN = {"logreg": "逻辑回归", "rf": "随机森林", "lgbm": "LightGBM", "xgb": "XGBoost"}


def balanced_pipeline(model_name: str, strategy: str = "none",
                      seed: int = SEED, linear: bool | None = None, **kwargs):
    """构造"预处理 + 不平衡处理 + 分类器"的完整管线。

    strategy : none / class_weight / smote / undersample
        smote 与 undersample 仅在训练数据上执行（imblearn Pipeline 保证）。
    """
    name = model_name.lower()
    if linear is None:
        linear = name == "logreg"

    if strategy == "class_weight" and name in ("logreg", "rf", "lgbm"):
        kwargs.setdefault("class_weight", "balanced")

    clf = build_model(name, seed=seed, **kwargs)
    steps = []
    if linear:
        steps.append(("prep", make_preprocessor([], scale=True)))
    if strategy in ("smote", "undersample"):
        if ImbPipeline is None:
            raise ImportError("需要 imbalanced-learn")
        sampler = (SMOTE(random_state=seed, k_neighbors=5)
                   if strategy == "smote"
                   else RandomUnderSampler(random_state=seed))
        steps.append(("sampler", sampler))
        return ImbPipeline(steps + [("clf", clf)])
    if steps:
        return Pipeline(steps + [("clf", clf)])
    return Pipeline([("clf", clf)])


def fit_pipeline(pipe, X_train, y_train, feature_names, strategy="none"):
    """训练管线；线性模型在此注入真实特征列名（预处理需要知道列名）。"""
    named = getattr(pipe, "named_steps", {})
    if "clf" in named and named["clf"].__class__.__name__ == "XGBClassifier" \
            and strategy == "class_weight":
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        pipe.set_params(clf__scale_pos_weight=neg / max(pos, 1.0))
    if "prep" in named:
        pipe.set_params(prep=make_preprocessor(list(feature_names), scale=True))
    pipe.fit(X_train, y_train)
    return pipe