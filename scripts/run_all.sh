#!/usr/bin/env bash
# 一键复现实验：数据 -> 训练 -> 评估 -> 图表表格
set -euo pipefail

cd "$(dirname "$0")/.."

STRATEGY="${STRATEGY:-none}"

echo "== 1/4 获取数据 =="
python scripts/download_data.py

echo "== 2/4 训练（多模型交叉验证，不平衡策略=${STRATEGY}） =="
python -m src.train --model all --strategy "$STRATEGY"

echo "== 3/4 评估（代价敏感阈值） =="
for m in logreg rf gbdt; do
  python -m src.evaluate --model "$m"
done

echo "== 4/4 完成 =="
echo "图表见 results/figures/，表格见 results/tables/，指标见 results/metrics/"
