# -*- coding: utf-8 -*-
"""可复现性工具：随机种子、计时、结果落盘、绘图风格。"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np

from .config import (
    FIG_DIR,
    FIG_FORMAT,
    METRIC_DIR,
    PLOT_STYLE,
    SEED,
    TABLE_DIR,
)


def set_seed(seed: int = SEED) -> None:
    """固定 python / numpy 随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer(label: str = ""):
    """``with timer("训练"):`` → 结束后打印耗时。"""
    t0 = time.time()
    yield
    print(f"[耗时] {label}: {time.time() - t0:.2f} s")


def run_tag(model: str, strategy: str = "none",
            feature_set: str = "engineered", seed: int = SEED) -> str:
    """统一的实验标识：模型__策略__特征集__种子。"""
    return f"{model}__{strategy}__{feature_set}__seed{seed}"


def save_metrics(metrics: dict, name: str) -> None:
    """指标写入 ``results/metrics/<name>.json``。"""
    path = METRIC_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    print(f"[指标] {path}")


def save_table(df, name: str) -> None:
    """表格同时输出 csv（留档）与 markdown（直接粘进报告）。"""
    csv_path = TABLE_DIR / f"{name}.csv"
    md_path = TABLE_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(f"[表格] {csv_path} / {md_path}")


def save_fig(fig, name: str):
    """图形保存到 ``results/figures/<name>.<fmt>``。"""
    path = FIG_DIR / f"{name}.{FIG_FORMAT}"
    fig.savefig(path, dpi=PLOT_STYLE["savefig.dpi"], bbox_inches="tight")
    print(f"[图] {path}")
    return path


def setup_cjk_font() -> str:
    """注册单个中文字体文件（Noto CJK 以 TTC 集合发布，matplotlib 无法直接索引）。"""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    from .config import ROOT

    font_path = ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        name = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        name = "Noto Sans CJK SC"
    plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return name


def apply_plot_style() -> None:
    """统一 matplotlib 风格（含中文字体与负号显示）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PLOT_STYLE)
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
