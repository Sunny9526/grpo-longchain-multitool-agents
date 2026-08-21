# GRPO Training and Optimization for Long-Chain Multi-Tool Agents

本目录集中保存该项目的实验记录、结果与工程说明。主实验代码位于 [`grpo-longchain-multitool-agents/`](../grpo-longchain-multitool-agents/)。

## 推荐阅读顺序

1. [Vanilla GRPO 崩溃诊断](experiments/vanilla-grpo/vanilla_grpo_diagnosis.md)：奖励饱和、训练集偏置和逐轮推理退化。
2. [消融实验设计](experiments/ablation/ablation_plan.md)：Turn-Discount、LATA、PRM-Lite 与联合方案的设计。
3. [消融诊断报告](experiments/ablation/ablation_diagnosis_report.md)：逐 checkpoint 结果、机制分析与局限。
4. [分布式训练部署](engineering/distributed_training_deployment.md)：两卡主消融与三卡优化拓扑的口径。

## 实验与结果

- [`experiments/ablation/`](experiments/ablation/)
  - [完整消融报告](experiments/ablation/ablation_diagnosis_report.md)
  - [实验设计手册](experiments/ablation/ablation_plan.md)
  - [机器可读结果](experiments/ablation/results_summary.csv)
- [`experiments/vanilla-grpo/`](experiments/vanilla-grpo/)
  - [Vanilla 崩溃诊断](experiments/vanilla-grpo/vanilla_grpo_diagnosis.md)
  - [Vanilla 训练优化](experiments/vanilla-grpo/vanilla_grpo_optimization.md)

## 工程文档

- [veRL rollout 设计](engineering/verl_rollout_design.md)
- [SFT 数据采集优化](engineering/collect_sft_optimization.md)
- [分布式训练部署](engineering/distributed_training_deployment.md)
- [上游代码与项目修改边界](engineering/upstream-modifications.md)
- [技术复盘](retrospective.md)

## 图表复现

图表统一存放于 [`assets/figures/`](assets/figures/)，由以下命令从报告转录的 CSV 生成：

```bash
python grpo-longchain-multitool-agents/scripts/plotting/plot_ablation_figures.py
```

这些图复现的是报告中保存的汇总值。主实验原始 `eval_report.json`、`split_eval_report.json` 和训练日志未随仓库发布，因此不能将其视为重新执行评测的结果。
