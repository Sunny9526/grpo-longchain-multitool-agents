# 上游代码与项目修改边界

本仓库为便于复现实验，直接包含 veRL 与 τ-bench。它们不是本项目从零实现的组件，仍分别遵循各自的许可证。

## 本项目原创部分

- `grpo-longchain-multitool-agents/src/envs/tau_bench_interaction.py`
  - τ-bench 交互适配；
  - PRM-Lite 逐轮过程奖励与规则；
  - assistant 内容记录和奖励组合。
- `grpo-longchain-multitool-agents/src/evaluation/`
  - 独立 pass@k 评测与结果汇总。
- `grpo-longchain-multitool-agents/scripts/` 与 `configs/`
  - SFT 数据采集、GRPO 数据构建、实验启动与评测配置。

## veRL fork 中的主要修改

- `verl/verl/trainer/ppo/core_algos.py`
  - Turn-Discounted Advantage；
  - LATA 长度感知优势归一化；
  - 项目所需的 advantage estimator 注册。
- `verl/verl/experimental/agent_loop/`
  - 将 τ-bench 多轮工具交互接入 rollout；
  - 记录逐轮 assistant 内容，供 PRM-Lite 计算过程奖励。

其中 agent loop 当前直接导入项目 `src`，因此 vendored veRL 不能脱离本项目独立使用。为减少训练主链路的回归风险，本次公开整理保留该耦合；如果后续长期跟进 veRL 上游，建议改为回调或插件接口。

## τ-bench

`tau-bench/` 保留上游 benchmark 代码与许可证。项目通过自身 wrapper、配置和评测脚本使用 airline 环境，不将 benchmark 本身声明为个人贡献。

## 版本说明

- veRL：仓库内版本 `0.6.1`
- τ-bench：以本仓库 vendored revision 为准
- 策略模型与用户模拟器：Qwen2.5 系列，遵循模型发布方条款
