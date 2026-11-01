#!/usr/bin/env bash
# 一键复现：数据获取 -> 主线实验 -> 补充分析
# 用法：bash scripts/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 1/4 获取数据 =="
python scripts/download_data.py

echo "== 2/4 主线实验（多模型 × 多不平衡策略 × 5 种子） =="
python scripts/run_experiments.py

echo "== 3/4 补充分析：校准与不平衡策略 =="
python scripts/extra_calibration_imbalance.py

echo "== 4/4 补充分析：错误个案 =="
python scripts/extra_error_analysis.py

echo
echo "完成。产物："
echo "  results/tables/   结果表（csv + md）"
echo "  results/figures/  插图（300 dpi）"
echo "  results/metrics/  指标 json"
