# 分布式训练部署与运行原理

> 面向项目复现与架构审阅的部署文档。  
> 依据以**启动脚本 + veRL 源码**为准；仓库内历史文档口径不一致时，以本文「权威证据」一节为准。  
> 整理日期：2026-07-21

---

## 0. 架构概览

本项目在 **2×A800** 上做长链路 Agent GRPO 时：

- **GPU0**：7B 策略侧 HybridEngine  
  - FSDP Actor / Ref（LoRA）  
  - vLLM 异步多轮 rollout  
  - 靠 `free_cache_engine` 的 sleep/wake + FSDP `param_offload` **时分复用**，峰值不错峰叠加
- **GPU1**：72B-AWQ **用户模拟器**，独立 vLLM OpenAI 服务（默认 TP=1），只在 rollout 多轮里被 HTTP 调用
- **不是**「2 卡上同时常驻 4 套满血模型」  
- Vanilla 若开 **rollout TP=2**，策略侧占满 2 卡，用户模拟器需 **第 3 张卡**（那是优化路径，不是消融主结果口径）

---

## 1. 硬件拓扑：2 卡 vs 3 卡（权威证据）

### 1.1 消融主结果 = 2 卡（简历应对齐这个）

| 证据 | 内容 |
|------|------|
| `scripts/train/grpo/run_exp{1,2,3,4}_*.sh` | `CUDA_VISIBLE_DEVICES=0` |
| 对应 yaml（如 `prm_lite_lata.yaml`） | `n_gpus_per_node: 1`，`tensor_model_parallel_size: 1` |
| `scripts/vllm_server/72b.sh` | 默认 `CUDA_DEVICES=1`，`TP_SIZE=1`，`PORT=8001` |
| `configs/interaction_config/*.yaml` | `user_base_url: http://localhost:8001/v1` |
| `docs/experiments/ablation/ablation_diagnosis_report.md` | 「2×A800（GPU0 7B policy, GPU1 72B user sim）」 |

```text
物理 GPU0  ← 训练进程可见 cuda:0（7B Hybrid）
物理 GPU1  ← 独立进程跑 72B vLLM（用户模拟器）
```

### 1.2 Vanilla TP=2 优化路径 = 3 卡

| 证据 | 内容 |
|------|------|
| `scripts/train/grpo/run_vanilla.sh` | `CUDA_VISIBLE_DEVICES=0,1` |
| `configs/train/grpo/vanilla_grpo.yaml` | `n_gpus_per_node: 2`，`tensor_model_parallel_size: 2` |
| `docs/experiments/vanilla-grpo/vanilla_grpo_optimization.md` | 标题写明 A800×3；GPU0+1=Policy(TP=2)，GPU2=72B |

```text
物理 GPU0+1 ← 7B 策略 FSDP + vLLM TP=2
物理 GPU2   ← 72B 用户模拟器
```

**口径约束**：不能将「2×A800 + rollout TP=2 + 72B TP=2」描述为同一实验拓扑，物理卡数对不上。

### 1.3 四个“组件”如何理解

| 名称 | 是否独占一张卡 | 实际含义 |
|------|----------------|----------|
| 7B Actor（FSDP） | 否 | 与 rollout/ref 同策略卡，训练阶段占用 |
| 7B Ref（FSDP） | 否 | LoRA 下多为同一 PeftModel 关 adapter；同卡 |
| 7B Rollout（vLLM） | 否 | 同策略卡，采样阶段占用，训练时 sleep |
| 72B User Sim（vLLM） | **是** | 独立服务，常驻另一张卡 |

---

## 2. 角色与 HybridEngine 原理

### 2.1 三个策略侧角色

| 角色 | 职责 | 引擎 |
|------|------|------|
| **Actor** | 当前策略前向 + 反传更新 LoRA | FSDP |
| **Ref** | 参考策略 logprob，做 KL | FSDP（LoRA 下 = base，关 adapter） |
| **Rollout** | 多轮 tool-call 采样 | vLLM async |

veRL 强制 HybridEngine：Actor 与 Rollout **语义上共用同一套策略权重**，阶段切换时做权重同步。

```python
# verl/trainer/ppo/ray_trainer.py
self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
assert self.hybrid_engine, "Currently, only support hybrid engine"
```

### 2.2 为何能塞进一张策略卡

核心约束：**推理峰值（KV）与训练峰值（激活/梯度）不能同时出现。**

- Rollout：KV cache 主导  
- Update：激活 + 梯度主导  
- 手段：`free_cache_engine` + FSDP offload +（本项目）`bypass_mode` 跳过巨大 logits 重算

---

## 3. 一个 Training Step 的代码运作流程

主循环：`verl/trainer/ppo/ray_trainer.py`

```text
① gen（rollout 采样）
② reward（τ-bench outcome / PRM-Lite；72B 在另一张卡）
③ old_log_prob
     └─ bypass_mode=true → 直接用 rollout_log_probs，跳过 FSDP 重算
④ ref_log_prob
⑤ advantage（GRPO / LATA 等）
⑥ update_actor
```

### 3.1 阶段 A：Rollout（采样）

入口（async）：

```python
# agent_loop.py · AgentLoopManager.generate_sequences
if free_cache_engine:
    self.wake_up()          # → worker.rollout_mode()
# 多 worker 并发 tool_agent 多轮生成
if free_cache_engine:
    self.sleep()            # → worker.trainer_mode()
```

`AsyncActorRolloutRefWorker`：

```python
async def wake_up(self):  await self.rollout_mode()
async def sleep(self):    await self.trainer_mode()
```

**`rollout_mode()` 关键顺序（`fsdp_workers.py`）**：

1. `load_fsdp_model_to_gpu`（若 `param_offload`）
2. 从 Peft/FSDP **导出**当前权重（LoRA 或整模）
3. `offload_fsdp_model_to_cpu` ← **采样期间 FSDP 不占 GPU**
4. `rollout.resume(tags=["weights"])` ← 唤醒 vLLM 权重区
5. `update_weights(...)` ← 灌入最新 LoRA（必要时含 base）
6. `resume(tags=["kv_cache"])` ← 开 KV，开始生成
7. 多轮中：policy 在本卡 vLLM；用户回复 HTTP 打到 **GPU1:8001** 的 72B

### 3.2 阶段 B：Reward

- Interaction（`TauBenchInteraction`）驱动 env
- 用户模拟器：独立 vLLM，**不参与** sleep/wake 切换
- 策略卡此时通常已 `sleep`，峰值在用户模拟器侧或 CPU 侧逻辑

### 3.3 阶段 C：old_log_probs（bypass）

```python
# rollout_corr_helper.py · apply_rollout_correction
batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
```

含义：

- `old_log_probs` 是 **log 概率张量**，不是一份 LoRA 参数
- 锚点 = **本 batch 采样时**的 `π_rollout`（当时的 LoRA_t）
- PPO ratio = `π_θ / π_rollout`（不是经典的 `π_θ / π_old_recompute`）

**常见误区**：以为每个 micro-batch 都要重算「更新前」的 old。  
正确语义：对本批数据 **冻住一次**；后续 mini/micro/epoch 只更新当前 `π_θ`，锚点不变。

### 3.4 阶段 D：Ref logprob

本项目 `lora_rank=16` → `ref_in_actor=True`：

```python
# ray_trainer.py
self.ref_in_actor = (lora_rank > 0 or lora_adapter_path is not None)
# 计算时走 actor_rollout_wg.compute_ref_log_prob
```

```python
# fsdp_workers.py · compute_log_prob
adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
with adapter_ctx:
    output, entropys = self.actor.compute_log_prob(...)
```

- Ref ≈ **同一份 FSDP PeftModel，关掉 LoRA adapter 的 base 前向**
- `ref_in_actor` 时 **不会** 对独立 `RefPolicy` worker 做 `init_model()`，避免再常驻一整份 7B
- 用途是 **KL**，不要和 `old_log_probs` 混为一谈

### 3.5 阶段 E：Update Actor

```python
# fsdp_workers.py · update_actor
load_fsdp_model_to_gpu(...)   # 若 offload
# update_policy：前向 + 反传 + optimizer
offload_fsdp_model_to_cpu(...)
```

配合：

- `use_fused_kernels=true`：fused CE，避免物化完整 `[B,L,V]` logits
- `bypass_mode`：已跳过 actor 侧 `compute_log_prob` 的巨大峰值

更新后得到 LoRA_{t+1}；**下一步**采样再 sync 进 vLLM。

---

## 4. LoRA / 概率锚点：一张符号时间线

```text
Step t:
  同步 LoRA_t → vLLM
  采样 a ~ π_{LoRA_t}
  记录 old_log_probs = log π_{LoRA_t}(a|s)   # bypass
  计算 ref_log_prob = log π_base(a|s)        # disable_adapter
  更新 LoRA_t → LoRA_{t+1}                   # ratio = π_θ / π_rollout

Step t+1:
  同步 LoRA_{t+1} → vLLM
  再采样……
```

| 名字 | 是什么 | 用途 |
|------|--------|------|
| LoRA 参数 | 可训练权重 | 策略本体 |
| `rollout_log_probs` / `old_log_probs` | 数值张量 | PPO/GRPO 重要性比的分母 |
| `ref_log_prob` | 数值张量 | KL 正则 |

---

## 5. 显存交接：FSDP ↔ vLLM（易混点）

### 5.1 不是「对称 CPU offload」

| 侧 | 非活跃时 | 机制 |
|----|----------|------|
| FSDP | 采样前导出后 **CPU offload** | `param_offload` + `offload_fsdp_model_to_cpu` |
| vLLM | 训练阶段 **`sleep` / `release`** | vLLM 自带 sleep，不是把同一块权重搬给 FSDP |

逻辑上共用「同一套 base+LoRA 语义」；**物理上** FSDP 与 vLLM 是两套引擎内存，靠 sync 对齐。

### 5.2 `sleep_level` 决定 vLLM「睡多死」

```python
# verl/third_party/vllm/__init__.py
# vLLM >= 0.8.5 → VLLM_SLEEP_LEVEL = 2
```

| level | 训练阶段 vLLM | 下次采样 |
|-------|---------------|----------|
| 1 | 主要释 KV；权重可能仍在 | 主要更新 LoRA |
| 2 | **权重也会被销毁**（代码注释明确） | **重灌 base + 挂最新 LoRA** |

`fsdp_workers.rollout_mode` 对 level=2 + LoRA 有专门分支：分别 `update_weights` base 与 LoRA。

### 5.3 一张交接图

```text
【采样】
  FSDP: load → 导出 → CPU offload
  vLLM: wake(weights) → 灌 base?/LoRA → wake(kv) → 生成

【训练】
  vLLM: sleep/release
  FSDP: load → ref / update → offload

【下步采样】
  再用新 LoRA（level2 还要重灌 base）唤醒 vLLM
```

---

## 6. 关键配置速查（消融主结果口径）

```yaml
# 策略侧（单卡）
trainer.n_gpus_per_node: 1
actor_rollout_ref.rollout.name: vllm
actor_rollout_ref.rollout.mode: async
actor_rollout_ref.rollout.tensor_model_parallel_size: 1
actor_rollout_ref.rollout.free_cache_engine: true
actor_rollout_ref.rollout.calculate_log_probs: true
actor_rollout_ref.model.lora_rank: 16
actor_rollout_ref.actor.fsdp_config.param_offload: true
actor_rollout_ref.actor.fsdp_config.optimizer_offload: true
actor_rollout_ref.actor.use_fused_kernels: true
algorithm.rollout_correction.bypass_mode: true

# 用户模拟器（另一进程）
user_base_url: http://localhost:8001/v1
# 72b.sh: CUDA_DEVICES=1, TP_SIZE=1
```

启动关系：

```bash
# 终端 1：用户模拟器
CUDA_DEVICES=1 PORT=8001 bash scripts/vllm_server/72b.sh

# 终端 2：训练（只看见 1 张策略卡）
export CUDA_VISIBLE_DEVICES=0
bash scripts/train/grpo/run_exp4_prm_lite_lata.sh
```

---

## 7. 关键源码索引

| 主题 | 路径 |
|------|------|
| 训练主循环 | `verl/verl/trainer/ppo/ray_trainer.py` |
| bypass old_log_probs | `verl/verl/trainer/ppo/rollout_corr_helper.py` → `apply_rollout_correction` |
| wake/sleep 封装 | `verl/verl/experimental/agent_loop/agent_loop.py` |
| rollout_mode / trainer_mode | `verl/verl/workers/fsdp_workers.py` |
| FSDP↔vLLM sharding | `verl/verl/workers/sharding_manager/fsdp_vllm.py` |
| disable_adapter 做 Ref | `fsdp_workers.py` → `compute_log_prob` / `compute_ref_log_prob` |
| Actor 更新读冻住的 old | `verl/verl/workers/actor/dp_actor.py` |
| 72B 服务 | `scripts/vllm_server/72b.sh` |
| Interaction | `configs/interaction_config/tau_bench_airline*.yaml` |

---

## 8. 准确描述与常见误区

### 准确描述

> 分布式训练部署：2×A800 分离部署——GPU0 运行 7B 策略（FSDP Actor/Ref + vLLM 异步多轮 rollout，`free_cache_engine` 时分复用），GPU1 运行 72B-AWQ 用户模拟器（独立 vLLM 服务）；结合 bypass 与算子融合控制显存，实现 2 卡稳定训练。

### 不准确描述

- 「2×A800 上 7B FSDP + rollout TP=2 + Ref TP + 72B TP=2」  
- 将不同硬件拓扑、训练配置或未重新执行的报告结果混为同一次复现实验  
- 把 Ref 说成第三张卡上的独立大模型  

### FAQ

**Q：2 卡怎么塞下 Actor、Ref、Rollout、72B？**  
A：72B 独占一卡；策略三角色在另一卡 Hybrid 时分复用——采样 wake vLLM、训练 sleep vLLM 并 FSDP offload；LoRA 下 Ref 是同一 PeftModel 关 adapter，不是第三份常驻 7B。

**Q：old_log_probs 从哪来？**  
A：bypass 下等于采样时 vLLM 的 `rollout_log_probs`，对本 batch 冻住；更新用 `π_θ/π_rollout`。

**Q：vLLM 和 FSDP 是互相 CPU offload 吗？**  
A：不对称。FSDP 是 `param_offload`；vLLM 是 `sleep`。新版本 sleep_level=2 会拆掉权重，下轮从 FSDP 重灌 base+LoRA。

---

## 9. 文档冲突备忘（复习时别被带偏）

| 文档 | 说法 | 如何处理 |
|------|------|----------|
| README / 消融报告 | 2×A800，GPU0 policy / GPU1 user sim | **主结果口径，优先** |
| `vanilla_grpo_optimization.md` | A800×3，TP=2 | Vanilla 优化路径，单独记 |
| `verl_rollout_design.md` 早期 | 2 卡设计草案 | 历史设计，以最终脚本为准 |
| 启动脚本 `CUDA_VISIBLE_DEVICES` | Exp=0；Vanilla=0,1 | **最快仲裁证据** |

---

## 10. 部署自检清单

- [ ] 能画出 2 卡拓扑，并说明何时会变成 3 卡  
- [ ] 能按 gen→reward→bypass→ref→adv→update 背主循环  
- [ ] 能区分 `old_log_probs` / `ref_log_prob` / LoRA 参数  
- [ ] 能说明 sleep/wake 与 FSDP offload 不对称  
- [ ] 能说出 bypass、fused kernels、free_cache_engine 各自省的是哪块显存  
- [ ] 结果与硬件口径（2 卡 / TP=1）一致，不混入 Vanilla TP=2  

---

*部署细节以当前仓库脚本与 veRL 源码为准。*
