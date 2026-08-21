# Vanilla GRPO 实验

本目录记录 Vanilla GRPO 在 τ-bench airline 长链路工具任务上的失效模式和工程优化。

- [崩溃诊断](vanilla_grpo_diagnosis.md)：按 checkpoint 分析奖励饱和、训练集偏置和逐轮推理退化。
- [训练优化](vanilla_grpo_optimization.md)：面向三张 A800 的 Policy TP=2 + 独立 72B 模拟器方案。

主消融结果采用两卡拓扑，请勿将本目录的三卡优化配置与主结果硬件口径混用。
