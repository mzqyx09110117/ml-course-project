# -*- coding: utf-8 -*-
"""训练入口（命令行）。

用法::

    python -m src.train --model logreg
    python -m src.train --model all --strategy class_weight

流程约定：读取数据 → 分层划分 → 交叉验证（记录 PR-AUC / F1）→ 训练 → 保存模型与指标。
TODO(实验部分)：填充训练主体。
"""

from __future__ import annotations

import argparse

from .config import N_FOLDS, SEED, TEST_SIZE
from .models import MODEL_ZOO


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="信用违约预测：训练与交叉验证")
    ap.add_argument("--model", default="all", choices=[*MODEL_ZOO, "all"])
    ap.add_argument("--strategy", default="none",
                    choices=["none", "class_weight", "oversample", "undersample", "smote"])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[配置] model={args.model} strategy={args.strategy} "
          f"folds={args.folds} seed={args.seed} test_size={TEST_SIZE}")
    raise NotImplementedError("训练流程待实现：数据加载 → 交叉验证 → 模型与指标落盘")


if __name__ == "__main__":
    raise SystemExit(main())
