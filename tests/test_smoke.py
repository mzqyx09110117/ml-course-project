# -*- coding: utf-8 -*-
"""冒烟测试：保证包可导入、配置口径正确、接口契约明确。

运行::

    python tests/test_smoke.py
    # 或  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports():
    from src import config, data, evaluate, models, train, utils

    assert config.SEED == 42
    assert config.TARGET_COL == "default"
    assert callable(utils.set_seed)
    assert callable(utils.save_metrics)
    assert callable(data.build_dataset)
    assert callable(models.build_model)
    assert callable(train.main)
    assert callable(evaluate.main)


def test_config_consistency():
    """代价设置与阈值网格必须自洽，避免评估口径漂移。"""
    from src import config

    assert config.COST_FALSE_NEGATIVE > config.COST_FALSE_POSITIVE
    assert all(0.0 < t < 1.0 for t in config.THRESHOLD_GRID)
    assert 0.5 in config.THRESHOLD_GRID


def test_stub_contract():
    """未实现接口必须显式抛出 NotImplementedError，而不是静默返回 None。"""
    from src import data, models

    try:
        data.clean(None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("data.clean 应抛出 NotImplementedError")

    try:
        models.build_model("logreg")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("models.build_model 应抛出 NotImplementedError")


if __name__ == "__main__":
    test_imports()
    test_config_consistency()
    test_stub_contract()
    print("冒烟测试通过")
