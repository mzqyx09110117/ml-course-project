# -*- coding: utf-8 -*-
"""评估入口（命令行）。

用法::

    python -m src.evaluate --model logreg

输出约定：测试集指标 json、混淆矩阵 csv/markdown、代价敏感阈值曲线数据。
TODO(实验部分)：填充评估主体。
"""

from __future__ import annotations

import argparse

from .config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, THRESHOLD_GRID
from .models import MODEL_ZOO


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="信用违约预测：测试集评估")
    ap.add_argument("--model", default="logreg", choices=MODEL_ZOO)
    ap.add_argument("--threshold", type=float, default=None,
                    help="决策阈值；缺省时在 THRESHOLD_GRID 上按代价最小化搜索")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[配置] model={args.model} "
          f"cost(FN/FP)={COST_FALSE_NEGATIVE}/{COST_FALSE_POSITIVE} "
          f"网格点数={len(THRESHOLD_GRID)}")
    raise NotImplementedError("评估流程待实现：指标 → 混淆矩阵 → 阈值搜索 → 结果落盘")


if __name__ == "__main__":
    raise SystemExit(main())
