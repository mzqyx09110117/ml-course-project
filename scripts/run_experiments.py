#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键复现全部实验，产出 results/ 下的表格、指标与图。

运行::

    python scripts/run_experiments.py            # 全量
    python scripts/run_experiments.py --quick    # 单种子快速版（调试用）

产物约定
--------
- ``results/tables/tXX_*.csv``   报告表格的数据源（.md 版本供直接粘贴）
- ``results/metrics/*.json``     每次运行的完整指标
- ``results/figures/fXX_*.png``  300dpi 图片
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    FIG_DIR,
    FIG_FORMAT,
    METRIC_DIR,
    PLOT_STYLE,
    SEED,
    TABLE_DIR,
    TARGET_COL,
)
from src import data as D
from src.evaluate import compute_metrics, curve_data, ece_score
from src.models import MODEL_CN, MODEL_ZOO
from src.train import cross_validate, positive_scores, train_one

plt.rcParams.update(PLOT_STYLE)


def setup_cjk_font() -> str:
    """注册中文字体。

    Noto CJK 以 OTC 集合形式发布，matplotlib 无法直接索引其中的 SC 字面，
    因此首次运行时用 fontTools 抽出 SC 字面存成单个字体文件（不入库）。
    """
    import matplotlib.font_manager as fm

    font_path = ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
    if not font_path.exists():
        src = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        font_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from fontTools.ttLib import TTCollection
            coll = TTCollection(src, lazy=False)
            coll.fonts[2].save(font_path)  # index 2 = Noto Sans CJK SC
        except Exception as exc:  # 兜底：用系统自带的中文 TTF
            fallback = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
            if Path(fallback).exists():
                font_path = Path(fallback)
            else:
                raise RuntimeError("找不到可用的中文字体") from exc
    fm.fontManager.addfont(str(font_path))
    name = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return name


log_font = setup_cjk_font()

SEEDS = [42, 2024, 7, 123, 888]
RUNS: list[dict] = []
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def save_table(df: pd.DataFrame, name: str, caption: str = "") -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    md = df.to_markdown(index=False, floatfmt=".4f") if _has_tabulate() else df.to_csv(index=False)
    (TABLE_DIR / f"{name}.md").write_text(md, encoding="utf-8")
    log(f"表格 {name}  shape={df.shape}")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def save_fig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.{FIG_FORMAT}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"图片 {path.name}")


def record(res: dict) -> None:
    RUNS.append(res)
    (METRIC_DIR / "all_runs.json").write_text(
        json.dumps(RUNS, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 数据与 EDA
def stage_data():
    log("==== 数据加载与质量诊断 ====")
    raw = D.load_raw()
    cl = D.clean(raw)
    q = D.data_quality_report(raw, cl)
    (METRIC_DIR / "data_quality.json").write_text(
        json.dumps(q, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"质量报告: {q}")

    # 表 2-3 描述统计（原始字段）
    cols = ["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"]
    desc = cl[cols].describe().T.reset_index().rename(columns={"index": "字段"})
    desc.columns = ["字段", "样本数", "均值", "标准差", "最小值", "25%", "中位数", "75%", "最大值"]
    save_table(desc, "t03_descriptive", "主要数值字段描述统计")

    # 图 2-1 类别分布
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    cnt = cl[TARGET_COL].value_counts().sort_index()
    axes[0].bar(["正常还款(0)", "违约(1)"], cnt.values, color=["#5b8cff", "#f2705b"])
    for i, v in enumerate(cnt.values):
        axes[0].text(i, v, f"{v}\n({v / len(cl):.2%})", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("类别分布")
    axes[0].set_ylabel("样本数")
    axes[1].pie(cnt.values, labels=["正常还款", "违约"], autopct="%.2f%%",
                colors=["#5b8cff", "#f2705b"], startangle=90)
    axes[1].set_title("正类占比")
    save_fig(fig, "f01_target_distribution")

    # 图 2-2 违约率随关键字段变化
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    rate = cl.groupby("PAY_0")[TARGET_COL].agg(["mean", "size"]).reset_index()
    axes[0].bar(rate["PAY_0"].astype(str), rate["mean"], color="#5b8cff")
    axes[0].set_title("违约率 vs 最近一期还款状态 PAY_0")
    axes[0].set_xlabel("PAY_0（负数=正常，正数=逾期月数）")
    axes[0].set_ylabel("违约率")

    cl["UTIL_BIN"] = pd.cut(cl["BILL_AMT1"] / (cl["LIMIT_BAL"] + 1),
                            [-2, 0, .2, .5, .8, .95, 1.01, 10],
                            labels=["<=0", "0-20%", "20-50%", "50-80%", "80-95%", "95-100%", ">100%"])
    u = cl.groupby("UTIL_BIN", observed=True)[TARGET_COL].agg(["mean", "size"]).reset_index()
    axes[1].bar(u["UTIL_BIN"].astype(str), u["mean"], color="#f2a65b")
    axes[1].set_title("违约率 vs 信用利用率分箱")
    axes[1].tick_params(axis="x", rotation=30)

    cl["AGE_BIN"] = pd.cut(cl["AGE"], [20, 25, 30, 35, 40, 45, 50, 60, 80])
    a = cl.groupby("AGE_BIN", observed=True)[TARGET_COL].agg(["mean", "size"]).reset_index()
    axes[2].bar([str(x) for x in a["AGE_BIN"]], a["mean"], color="#7d5bf2")
    axes[2].set_title("违约率 vs 年龄分箱")
    axes[2].tick_params(axis="x", rotation=45)
    save_fig(fig, "f02_eda_target_rate")

    # 表 2-4 违约率分组统计（真实计数）
    grp = cl.groupby("PAY_0").agg(
        样本数=(TARGET_COL, "size"), 违约数=(TARGET_COL, "sum"), 违约率=(TARGET_COL, "mean")
    ).reset_index().rename(columns={"PAY_0": "最近一期还款状态"})
    save_table(grp, "t04_default_by_pay0", "按最近一期还款状态分组的违约率")

    # 图 2-3 关键数值特征分布（违约 vs 正常）
    fe = D.add_features(cl)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, col, title in zip(axes,
                              ["LIMIT_BAL", "BILL_AVG", "UTIL_AVG", "PAY_RATIO_MEAN"],
                              ["授信额度", "月均账单额", "平均信用利用率", "平均还款比"]):
        for lab, color in [(0, "#5b8cff"), (1, "#f2705b")]:
            vals = fe.loc[fe[TARGET_COL] == lab, col]
            vals = vals.clip(vals.quantile(.01), vals.quantile(.99))
            ax.hist(vals, bins=40, alpha=.6, label=f"{'正常' if lab == 0 else '违约'}", color=color)
        ax.set_title(title)
        ax.legend(fontsize=8)
    save_fig(fig, "f03_feature_distribution")

    # 图 2-4 特征相关性（与标签及相互间）
    num_cols = ["LIMIT_BAL", "AGE", "PAY_0", "BILL_AVG", "UTIL_AVG",
                "PAY_RATIO_MEAN", "LATE_CNT1", "DELAY_SUM", "PAY_AMT_AVG", TARGET_COL]
    corr = fe[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)), num_cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(num_cols)), num_cols, fontsize=8)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_title("关键特征相关系数矩阵")
    save_fig(fig, "f04_correlation")
    return cl


# ---------------------------------------------------------------- 主结果
def stage_main(X_tr, y_tr, X_va, y_va, X_te, y_te, seeds):
    log("==== 主结果：模型 × 不平衡策略 × 多随机种子 ====")
    combos = [(m, "none") for m in MODEL_ZOO] + [(m, "class_weight") for m in MODEL_ZOO] \
             + [("lgbm", "smote"), ("logreg", "smote")]
    rows, store = [], {}
    for model, strategy in combos:
        for sd in seeds:
            res, fitted, p_te, thr = train_one(
                model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                strategy=strategy, calibration="none", seed=sd)
            record(res)
            rows.append(res)
            if sd == seeds[0]:
                store[(model, strategy)] = (fitted, p_te, thr)
    df = pd.DataFrame(rows)
    agg = (df.groupby(["model", "strategy"])
             .agg(auc=("auc", "mean"), auc_std=("auc", "std"),
                  ap=("ap", "mean"), ap_std=("ap", "std"),
                  f1=("f1", "mean"), brier=("brier", "mean"), ece=("ece", "mean"),
                  ks=("ks", "mean"), cost=("expected_cost", "mean"),
                  cost05=("cost_at_05", "mean"), saving=("cost_saving", "mean"),
                  thr=("val_threshold", "mean"), n=("auc", "size"))
             .reset_index())
    agg["model_cn"] = agg["model"].map(MODEL_CN)
    agg["配置"] = agg["model_cn"] + " / " + agg["strategy"]
    save_table(agg[["配置", "auc", "auc_std", "ap", "ap_std", "f1", "ks",
                    "brier", "ece", "cost", "cost05", "saving", "thr", "n"]],
               "t05_main_results", "主结果：各模型与不平衡策略（5 种子均值）")

    # 稳定性表（种子间波动）
    stab = (df.groupby(["model", "strategy"])
              .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                   ap_mean=("ap", "mean"), ap_std=("ap", "std"),
                   cost_mean=("expected_cost", "mean"), cost_std=("expected_cost", "std"))
              .reset_index())
    save_table(stab, "t06_stability", "随机种子稳定性")

    # 图 5-1 ROC / 图 5-2 PR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model in MODEL_ZOO:
        fitted, p_te, _ = store[(model, "none")]
        c = curve_data(y_te, p_te)
        m = compute_metrics(y_te, p_te, 0.5)
        axes[0].plot(c["fpr"], c["tpr"], label=f"{MODEL_CN[model]} (AUC={m['auc']:.4f})")
        axes[1].plot(c["recall"], c["precision"], label=f"{MODEL_CN[model]} (AP={m['ap']:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8, label="随机猜测")
    axes[0].set_xlabel("假正率 FPR"); axes[0].set_ylabel("真正率 TPR")
    axes[0].set_title("ROC 曲线（测试集）"); axes[0].legend(fontsize=8)
    axes[1].axhline(y_te.mean(), ls="--", c="k", lw=.8, label=f"正类基线={y_te.mean():.3f}")
    axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
    axes[1].set_title("PR 曲线（测试集）"); axes[1].legend(fontsize=8)
    save_fig(fig, "f05_roc_pr")
    return store, df


# ---------------------------------------------------------------- 超参搜索
def stage_search(X_tr, y_tr):
    log("==== 超参搜索（5 折 CV，仅训练集） ====")
    rows = []
    for model, grid in [
        ("lgbm", [dict(num_leaves=nl, learning_rate=lr) for nl in (15, 31, 63) for lr in (0.03, 0.05, 0.1)]),
        ("xgb", [dict(max_depth=d, learning_rate=lr) for d in (3, 5, 7) for lr in (0.03, 0.05, 0.1)]),
    ]:
        best = None
        for params in grid:
            r = cross_validate(model, X_tr, y_tr, strategy="none", folds=5, **params)
            r.update({f"param_{k}": v for k, v in params.items()})
            rows.append(r)
            log(f"  {model} {params} → CV AUC={r['auc_mean']:.4f} AP={r['ap_mean']:.4f}")
            if best is None or r["ap_mean"] > best["ap_mean"]:
                best = r
        log(f"  ★ {model} 最优: {best['params'] if 'params' in best else [k for k in best if k.startswith('param_')]}")
    df = pd.DataFrame(rows)
    save_table(df, "t07_hyperparam_search", "超参数搜索结果（5 折 CV，按 AP 选择）")


# ---------------------------------------------------------------- 校准
def stage_calibration(X_tr, y_tr, X_va, y_va, X_te, y_te):
    log("==== 概率校准对比 ====")
    rows, probs = [], {}
    for model in ["logreg", "lgbm"]:
        for cal in ["none", "sigmoid", "isotonic"]:
            for sd in [SEED, 2024, 7]:
                res, fitted, p_te, thr = train_one(
                    model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                    strategy="none", calibration=cal, seed=sd)
                record(res)
                rows.append(res)
                if sd == SEED:
                    probs[(model, cal)] = p_te
    df = pd.DataFrame(rows)
    agg = (df.groupby(["model", "calibration"])
             .agg(auc=("auc", "mean"), ap=("ap", "mean"), brier=("brier", "mean"),
                  ece=("ece", "mean"), cost=("expected_cost", "mean"),
                  saving=("cost_saving", "mean"), n=("auc", "size")).reset_index())
    agg["模型"] = agg["model"].map(MODEL_CN)
    save_table(agg, "t08_calibration", "概率校准对比（Brier/ECE 为主）")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (model, cal), p in probs.items():
        ece, rows_b = ece_score(y_te, p)
        xs = [r["mean_pred"] for r in rows_b if r["count"] > 0]
        ys = [r["frac_pos"] for r in rows_b if r["count"] > 0]
        axes[0].plot(xs, ys, "o-", ms=3, lw=1,
                     label=f"{MODEL_CN[model]}-{cal} (ECE={ece:.4f})")
        axes[1].hist(p, bins=40, alpha=.5, label=f"{MODEL_CN[model]}-{cal}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8, label="完全校准参考线")
    axes[0].set_xlabel("预测违约概率"); axes[0].set_ylabel("实际违约频率")
    axes[0].set_title("可靠性曲线（测试集）"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("预测违约概率"); axes[1].set_ylabel("样本数")
    axes[1].set_title("预测概率分布"); axes[1].legend(fontsize=7)
    save_fig(fig, "f06_calibration")
    return probs


# ---------------------------------------------------------------- 代价敏感
def stage_cost(X_tr, y_tr, X_va, y_va, X_te, y_te):
    log("==== 代价敏感阈值决策 ====")
    rows, curves = [], {}
    for ratio in [(1, 5), (1, 10), (1, 20)]:
        c_fp, c_fn = float(ratio[0]), float(ratio[1])
        for cal in ["none", "isotonic"]:
            res, fitted, p_te, thr = train_one(
                "lgbm", X_tr, y_tr, X_va, y_va, X_te, y_te,
                strategy="none", calibration=cal, seed=SEED,
                c_fp=c_fp, c_fn=c_fn, use_cost_threshold=True)
            record(res)
            rows.append(res)
            curves[(ratio[1], cal)] = thr["cost_curve"]
    df = pd.DataFrame(rows)
    cols = ["cost_ratio", "calibration", "threshold", "threshold_analytic", "precision",
            "recall", "f1", "expected_cost", "cost_at_05", "cost_saving"]
    save_table(df[cols], "t09_cost_sensitivity", "误判代价比敏感性（LightGBM）")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (fn, cal), c in curves.items():
        axes[0].plot(c["threshold"], c["cost"], lw=1.4, label=f"FN:FP=1:{fn} / 校准={cal}")
    axes[0].set_xlabel("决策阈值"); axes[0].set_ylabel("测试集期望误判成本")
    axes[0].set_title("阈值—成本曲线"); axes[0].legend(fontsize=7); axes[0].set_yscale("log")

    d = df[df["calibration"] == "isotonic"]
    x = np.arange(len(d))
    axes[1].bar(x - .2, d["cost_at_05"], .4, label="固定阈值 0.5", color="#9aa4b8")
    axes[1].bar(x + .2, d["expected_cost"], .4, label="代价敏感阈值", color="#5b8cff")
    axes[1].set_xticks(x, [f"FN:FP={r}" for r in d["cost_ratio"]])
    axes[1].set_ylabel("期望误判成本"); axes[1].set_title("固定阈值 vs 代价敏感阈值")
    axes[1].legend(fontsize=8)
    save_fig(fig, "f07_threshold_cost")
    return df


# ---------------------------------------------------------------- 消融
def stage_ablation(cl, y_all):
    log("==== 消融实验 ====")
    rows = []

    def run_variant(tag, desc, feature_set, calibration, use_cost, strategy):
        feat = D.build_dataset(feature_set=feature_set)[0] if feature_set != "full" else X_FULL
        idx_tr, idx_va, idx_te = IDX_TR, IDX_VA, IDX_TE
        res, _, _, _ = train_one(
            "lgbm", feat.iloc[idx_tr], y_all[idx_tr], feat.iloc[idx_va], y_all[idx_va],
            feat.iloc[idx_te], y_all[idx_te], strategy=strategy,
            calibration=calibration, seed=SEED, use_cost_threshold=use_cost)
        res.update({"variant": tag, "description": desc})
        record(res)
        rows.append(res)
        log(f"  {tag}: AUC={res['auc']:.4f} ECE={res['ece']:.4f} cost={res['expected_cost']:.1f}")

    run_variant("full", "完整方法（特征工程+校准+代价敏感阈值）", "full", "isotonic", True, "none")
    run_variant("no_fe", "去掉特征工程（仅原始字段）", "raw", "isotonic", True, "none")
    run_variant("no_calib", "去掉概率校准", "full", "none", True, "none")
    run_variant("no_cost", "去掉代价敏感阈值（固定 0.5）", "full", "isotonic", False, "none")
    run_variant("with_imb", "额外加入 class_weight 类别加权（检验不平衡处理是否必要）", "full", "isotonic", True, "class_weight")
    run_variant("demo_only", "仅人口统计特征（冷启动场景）", "demo", "isotonic", True, "none")

    df = pd.DataFrame(rows)
    keep = ["variant", "description", "auc", "ap", "f1", "brier", "ece",
            "threshold", "expected_cost", "cost_saving"]
    save_table(df[keep], "t10_ablation", "消融实验（LightGBM，单种子）")

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    d = df.sort_values("expected_cost")
    ax.barh(d["variant"], d["expected_cost"], color="#5b8cff")
    for i, (v, c) in enumerate(zip(d["variant"], d["expected_cost"])):
        ax.text(c, i, f" {c:.0f}", va="center", fontsize=8)
    ax.set_xlabel("测试集期望误判成本（越低越好）")
    ax.set_title("消融实验：各变体的期望误判成本")
    save_fig(fig, "f08_ablation")
    return df


# ---------------------------------------------------------------- SHAP 与解释
def stage_shap(X_tr, y_tr, X_va, y_va, X_te, y_te):
    log("==== SHAP 可解释性 ====")
    import shap
    from src.models import balanced_pipeline
    from src.models import fit_pipeline

    pipe = balanced_pipeline("lgbm", strategy="none", seed=SEED)
    pipe = fit_pipeline(pipe, X_tr, y_tr, X_tr.columns)
    booster = pipe.named_steps["clf"]
    explainer = shap.TreeExplainer(booster)
    sample = X_te.sample(n=min(2000, len(X_te)), random_state=SEED)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]

    imp = pd.DataFrame({"feature": sample.columns,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False)
    save_table(imp.head(20), "t11_shap_top20", "SHAP 全局重要性 Top20")

    fig = plt.figure(figsize=(7.5, 5))
    shap.summary_plot(sv, sample, plot_type="bar", max_display=15, show=False)
    plt.title("SHAP 全局特征重要性（Top15）")
    save_fig(plt.gcf(), "f09_shap_bar")

    fig = plt.figure(figsize=(7.5, 6))
    shap.summary_plot(sv, sample, max_display=15, show=False)
    plt.title("SHAP 特征影响分布（Top15）")
    save_fig(plt.gcf(), "f10_shap_beeswarm")

    top = imp["feature"].head(3).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, f in zip(axes, top):
        j = list(sample.columns).index(f)
        ax.scatter(sample[f], sv[:, j], s=6, alpha=.45, c="#5b8cff")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel(f); ax.set_ylabel("SHAP 值")
        ax.set_title(f"{f} 的 SHAP 依赖")
    save_fig(fig, "f11_shap_dependence")

    # 混淆矩阵（完整方法）
    res, _, p_te, _ = train_one("lgbm", X_tr, y_tr, X_va, y_va, X_te, y_te,
                                strategy="none", calibration="isotonic", seed=SEED)
    record(res)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, thr, title in [(axes[0], 0.5, "固定阈值 0.5"),
                           (axes[1], res["threshold"], f"代价敏感阈值 {res['threshold']:.2f}")]:
        m = compute_metrics(y_te, p_te, thr)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["预测:正常", "预测:违约"])
        ax.set_yticks([0, 1], ["实际:正常", "实际:违约"])
        ax.set_title(f"{title}\n成本={m['expected_cost']:.0f}")
    save_fig(fig, "f12_confusion_matrix")
    return imp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="单种子快速模式")
    args = ap.parse_args()
    seeds = [SEED] if args.quick else SEEDS

    cl = stage_data()

    feat, y = D.build_dataset(feature_set="full")
    global X_FULL, IDX_TR, IDX_VA, IDX_TE
    X_FULL = feat
    IDX_TR, IDX_VA, IDX_TE = D.make_split(y, seed=SEED)
    X_tr, X_va, X_te = feat.iloc[IDX_TR], feat.iloc[IDX_VA], feat.iloc[IDX_TE]
    y_tr, y_va, y_te = y[IDX_TR], y[IDX_VA], y[IDX_TE]
    log(f"划分: train={len(X_tr)} val={len(X_va)} test={len(X_te)} 特征数={feat.shape[1]}")

    stage_main(X_tr, y_tr, X_va, y_va, X_te, y_te, seeds)
    if not args.quick:
        stage_search(X_tr, y_tr)
    stage_calibration(X_tr, y_tr, X_va, y_va, X_te, y_te)
    stage_cost(X_tr, y_tr, X_va, y_va, X_te, y_te)
    stage_ablation(cl, y)
    stage_shap(X_tr, y_tr, X_va, y_va, X_te, y_te)

    log(f"==== 全部完成，共 {len(RUNS)} 次运行，用时 {time.time() - T0:.1f}s ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())