# 信用违约风险预测 —— 面向类别不平衡与代价敏感决策

基于客户信用与还款记录的表格型二分类任务：预测借款人是否会发生违约（default），
并针对**类别不平衡**与**误判代价不对称**（把违约客户错判为正常客户的损失远大于相反情况）设计
重采样、阈值移动与代价敏感学习方案，在保证可解释性的前提下比较各模型的实用收益。

> **课程**：机器学习课程设计
> **作者**：祁震淼 · 20245627
> **班级**：人工智能 2403 班
> **专业**：人工智能
> **指导教师**：曹鹏

---

## 1. 问题定义

| 项目 | 内容 |
| --- | --- |
| 任务类型 | 表格型二分类（Tabular Binary Classification） |
| 标签 | `default`：是否违约（正类 = 违约） |
| 核心难点 | 正负样本比例失衡；两类错误的业务代价不对称 |
| 评价指标 | 以 **PR-AUC / F1 / Recall@固定精确率** 为主，Accuracy 仅作参考 |
| 决策方式 | 概率输出 + 代价敏感阈值选择，而非默认 0.5 |

> 在不平衡数据上，Accuracy 会给出"看起来很好但毫无用处"的结论（全预测为负类即可接近多数类占比）。
> 因此本项目的主指标不使用 Accuracy。

## 2. 数据集

| 项目 | 内容 |
| --- | --- |
| 名称 | 待补充（数据集定稿后填写） |
| 来源 | UCI Machine Learning Repository（具体数据集编号见 `scripts/download_data.py`） |
| 规模 | 待补充（样本数 × 特征数） |
| 正负比 | 待补充（违约样本占比） |
| 缺失情况 | 待补充 |

原始数据体积较大且可公开获取，**不纳入版本库**。获取方式：

```bash
python scripts/download_data.py     # 下载并写入 data/raw/
```

数据字段说明、下载校验与处理流程见 [`data/README.md`](data/README.md)。

## 3. 运行环境

- Python 3.13
- 依赖见 [`requirements.txt`](requirements.txt)（版本为实际运行环境快照，可直接复现）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 4. 目录结构

```text
.
├── src/                        # 核心代码包
│   ├── config.py               # 路径、随机种子、绘图与评估配置（唯一常量来源）
│   ├── data.py                 # 数据读取、清洗、特征/标签划分
│   ├── models.py               # 模型工厂与对比模型清单
│   ├── train.py                # 交叉验证 / 训练入口
│   ├── evaluate.py             # 指标计算、混淆矩阵、代价敏感阈值搜索
│   └── utils.py                # 随机种子、计时、结果落盘、绘图风格
├── scripts/
│   ├── download_data.py        # 原始数据获取 → data/raw/
│   └── run_all.sh              # 一键复现全流程
├── notebooks/                  # 探索性分析（提交前清空输出）
├── data/README.md              # 数据说明（原始/中间数据不入库）
├── results/
│   ├── figures/                # 报告插图（300 dpi）
│   ├── tables/                 # 结果表格（csv + markdown）
│   └── metrics/                # 指标 json（可重跑生成，不入库）
├── report/                     # 报告源文件与 PDF 构建脚本
├── tests/test_smoke.py         # 冒烟测试
├── requirements.txt
└── LICENSE
```

## 5. 复现步骤

```bash
# 一键复现（数据 → 训练 → 评估 → 图表）
bash scripts/run_all.sh

# 或分步执行
python scripts/download_data.py
python -m src.train --model all
python -m src.evaluate --model gbdt     # 模型名取自 src/models.py 的 MODEL_ZOO
```

所有随机过程统一由 `src/config.py` 中的 `SEED` 控制，同一环境重复运行结果一致。

## 6. 实验结果

> **状态：实验进行中。** 本节数据由 `results/tables/` 下的输出文件直接复制生成，
> 不在正文中手工填写，以保证报告与代码输出严格一致。

| 模型 | PR-AUC | F1（最优阈值） | Recall | 说明 |
| --- | --- | --- | --- | --- |
| 待补充 | 待补充 | 待补充 | 待补充 | 基线：逻辑回归 |
| 待补充 | 待补充 | 待补充 | 待补充 | 不平衡处理策略对比 |
| 待补充 | 待补充 | 待补充 | 待补充 | 代价敏感阈值决策 |

> 表格由 `results/tables/model_comparison.md` 生成后整表替换本节，
> 保留表头写法即可，避免手工誊抄出错。

## 7. 报告

课程设计报告的 Word 与 PDF 版本存放于 `report/` 目录（`report/report.docx`、`report/report.pdf`）。
由源文件重建 PDF：

```bash
bash report/build_pdf.sh tex      # 生成最终 PDF（矢量、字体嵌入）
bash report/build_pdf.sh layout   # 逐页核对分页与图表位置
```

## 8. 许可

本项目采用 [MIT License](LICENSE)。

## 9. 参考资料

1. UCI Machine Learning Repository：<https://archive.ics.uci.edu/>
2. scikit-learn 用户指南（模型评估与概率校准）：<https://scikit-learn.org/stable/user_guide.html>
3. 所选数据集的官方说明页与相关文献：待补充
