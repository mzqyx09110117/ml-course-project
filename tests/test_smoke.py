# -*- coding: utf-8 -*-
"""冒烟测试：核心模块可导入、接口契约稳定、合成数据全链路可跑通。

运行::

    python tests/test_smoke.py
    # 或  python -m pytest tests -q

设计原则
--------
不依赖 ``data/raw/`` 下的真实数据（体积大且不入库），全部用合成数据验证：
清洗 → 特征工程 → 划分 → 管线拟合 → 预测 → 指标计算。
任何一步接口变动都会在这里失败，避免"改了代码忘了改调用方"。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config


def _synthetic_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """构造与 UCI 信用卡数据同构的合成表（字段名与取值域一致）。"""
    rng = np.random.default_rng(seed)
    limit = rng.choice([20000, 50000, 100000, 200000], size=n).astype(float)
    data = {
        "LIMIT_BAL": limit,
        "SEX": rng.integers(1, 3, size=n),
        "EDUCATION": rng.choice([1, 2, 3, 0, 5, 6], size=n),   # 含未文档化取值
        "MARRIAGE": rng.choice([1, 2, 3, 0], size=n),          # 含未文档化取值
        "AGE": rng.integers(21, 70, size=n),
    }
    for i in range(1, 7):
        data[f"BILL_AMT{i}"] = rng.normal(limit * 0.7, limit * 0.3, size=n)
        data[f"PAY_AMT{i}"] = np.abs(rng.normal(limit * 0.1, limit * 0.2, size=n))
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        data[col] = rng.choice([-2, -1, 0, 1, 2, 3, 5], size=n)
    df = pd.DataFrame(data)
    # 让标签与"逾期次数"相关，保证模型能学到东西（AUC 明显高于 0.5）
    late = (df[["PAY_0", "PAY_2", "PAY_3"]].to_numpy() >= 1).sum(axis=1)
    prob = 1.0 / (1.0 + np.exp(-(late - 1.5)))
    df[config.TARGET_COL] = (rng.random(n) < prob).astype(int)
    return df


def test_imports():
    from src import data, evaluate, models, train, utils

    assert callable(data.clean)
    assert callable(data.add_features)
    assert callable(models.build_model)
    assert callable(models.balanced_pipeline)
    assert callable(models.fit_pipeline)
    assert callable(evaluate.compute_metrics)
    assert callable(train.train_one)
    assert callable(utils.set_seed)


def test_config_consistency():
    """代价设置与阈值网格必须自洽，避免评估口径漂移。"""
    assert config.SEED == 42
    assert config.TARGET_COL == "default"
    assert config.COST_FALSE_NEGATIVE > config.COST_FALSE_POSITIVE
    assert all(0.0 < t < 1.0 for t in config.THRESHOLD_GRID)
    assert 0.5 in config.THRESHOLD_GRID


def test_clean_and_features():
    from src import data

    df_raw = _synthetic_frame()
    df_clean = data.clean(df_raw)

    # 未文档化类别被归并
    assert set(df_clean["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(df_clean["MARRIAGE"].unique()).issubset({1, 2, 3})
    assert df_clean[config.TARGET_COL].dtype.kind in "iu"
    # 特征工程必须新增列且不丢原列
    feat = data.add_features(df_clean)
    assert feat.shape[1] > df_clean.shape[1]
    assert feat[["BILL_AVG", "UTIL_CUR", "PAY_RATIO_MEAN", "LATE_CNT1"]].notna().all().all()
    assert "LIMIT_BAL" in feat.columns


def test_split_is_stratified():
    from src import data

    y = _synthetic_frame().to_numpy()[:, -1].astype(int)
    tr, va, te = data.make_split(y)
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0
    assert abs(len(tr) / len(y) - 0.7) < 0.02
    # 各子集正类比例应接近总体比例
    base = y.mean()
    for idx in (tr, va, te):
        assert abs(y[idx].mean() - base) < 0.05


def test_unknown_model_raises():
    from src import models

    try:
        models.build_model("no_such_model")
    except ValueError:
        pass
    else:
        raise AssertionError("未知模型应抛出 ValueError")


def test_pipeline_end_to_end():
    """合成数据上跑通 预处理 → 拟合 → 预测 → 指标。"""
    from src import data, evaluate, models

    df = data.add_features(data.clean(_synthetic_frame(n=300)))
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL].to_numpy().astype(int)

    pipe = models.balanced_pipeline("logreg", strategy="none", seed=config.SEED)
    pipe = models.fit_pipeline(pipe, X, y, list(X.columns), strategy="none")
    prob = pipe.predict_proba(X)[:, 1]

    assert prob.shape == y.shape
    assert np.all((prob >= 0) & (prob <= 1))

    metrics = evaluate.compute_metrics(y, prob, threshold=0.5)
    for key in ("auc", "ap", "f1", "ks", "brier", "ece", "expected_cost"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert metrics["auc"] > 0.5          # 合成数据有信号，模型应能学到


if __name__ == "__main__":
    test_imports()
    test_config_consistency()
    test_clean_and_features()
    test_split_is_stratified()
    test_unknown_model_raises()
    test_pipeline_end_to_end()
    print("冒烟测试通过（6 项）")
