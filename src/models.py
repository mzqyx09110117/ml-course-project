# -*- coding: utf-8 -*-
"""模型工厂与对比模型清单。

新增对比模型时只在本文件注册，训练/评估脚本无需改动。
"""

from __future__ import annotations

from .config import SEED

# 报告中的模型对比顺序以此列表为准，保证"表格顺序 = 代码顺序"
MODEL_ZOO = [
    "logreg",        # 基线：逻辑回归（需标准化）
    "rf",            # 随机森林：非线性 + 特征重要性可解释
    "gbdt",          # 梯度提升树：表格任务的强基线
]


def build_model(name: str, **kwargs):
    """按名称构造模型（不平衡处理与代价敏感选项随模型一同注入）。

    Parameters
    ----------
    name : str
        ``MODEL_ZOO`` 中的名称。
    **kwargs
        覆盖默认超参数。

    TODO(实验部分)：返回具体估计器；未知名称抛出 ``ValueError``。
    """
    raise NotImplementedError(f"build_model({name!r}) 待实现")


def balanced_pipeline(name: str, strategy: str = "none", **kwargs):
    """在 ``build_model`` 之上叠加不平衡处理策略。

    Parameters
    ----------
    strategy : {"none", "class_weight", "oversample", "undersample", "smote"}
        不平衡处理方式，需与 ``config`` 中的代价设置保持一致的评估口径。

    TODO(实验部分)
    """
    raise NotImplementedError(f"balanced_pipeline({name!r}, {strategy!r}) 待实现")
