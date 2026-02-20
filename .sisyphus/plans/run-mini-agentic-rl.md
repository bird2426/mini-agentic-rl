# mini-agentic-rl: 运行 agent-lightning mini 版 (Mini-GRPO + Mini-LLM-Server)

## TL;DR

> **目标**: 在本机 (4GB 显存) 运行 mini-agentic-rl，实现 GRPO 训练流程
> 
> **核心架构**:
> - **Trainer**: 编排训练循环 (Agent-Lightning)
> - **Runner**: 收集轨迹 (无感 Trace)
> - **Algorithm**: **Mini-GRPO** (自定义 PyTorch + LoRA，替代 verl)
> - **LLM Backend**: **Mini-LLM-Server** (自定义 FastAPI + Transformers + LoRA，替代 vllm)
> 
> **关键特性**:
> - **无感 Trace**: Agent 通过 OpenAI SDK 调用 Mini-LLM-Server，保留完整 Trace 能力
> - **低显存**: 使用 **LoRA** 只训练 1-2% 参数，实测 4GB VRAM 可运行 Qwen2-0.5B
> 
> **交付物**: 
> - 可运行的 Mini-GRPO 训练脚本
> - MiniGRPO 算法实现 (集成 LoRA)
> - Mini-LLM-Server 实现
> - 保留 Agent-Lightning Hooks 机制
> 
> **预计工作量**: Medium
> **并行执行**: YES - 5 waves
> **关键路径**: Import 修复 → 显存优化设计 → Mini-LLM-Server → Mini-GRPO(LoRA) → 训练验证 → Benchmark SOTA

---

## 当前进度 (已完成)

✅ **Wave 1**: Import 修复完成
✅ **Wave 2**: config.py + emitter/ 已创建
✅ **Wave 3**: Mini-LLM-Server 已创建
✅ **Wave 4**: Mini-GRPO + LoRA 已实现
✅ **Wave 5**: Agent-Lightning 框架集成完成
✅ **Wave 6**: 代码清理 + 正确使用示例
⏳ **Wave 7**: 完整训练集成 (正确的 Agent-Lighting 方式)
⏳ **Wave 8**: Legacy 代码清理 (未来)

### 实测验证结果

```
模型: Qwen2-0.5B + LoRA
显存: 4GB VRAM (RTX 3050)
训练参数: 4,399,104 / 498,431,872 = 0.88%
状态: 训练成功，无 OOM

验证结果 (examples/run_grpo.py):
- 10 iterations 训练
- 从 0/4 提升到 4/4 正确率
- Loss 正常收敛

验证结果 (examples/train_with_trainer.py):
- 5 iterations 训练完成
- Agent-Lightning 集成模式正常工作
```

---

## Work Objectives

### 核心目标
实现一个可运行的 mini-agentic-rl，支持 RL/GRPO 训练流程

### 必须包含 (Must Have)
- [x] 所有 import 语句修复 (`agentlightning` → `mini_agent_lighting`)
- [x] config.py 配置模块
- [x] emitter/ 事件发射模块
- [x] **Mini-LLM-Server**: 兼容 OpenAI API 的轻量级服务
- [x] **Mini-GRPO**: 纯 PyTorch 实现的 GRPO 算法 **+ LoRA 支持**
- [x] 能运行训练流程
- [x] **集成 Agent-Lightning 框架**: 保留 Trainer + Runner + Hooks

### 必须不包含 (Must NOT)
- [ ] verl/ - 完全删除 (用 Mini-GRPO 替代)
- [ ] instrumentation/ - 完全删除 (用 Mini-LLM-Server + LLMProxy 替代)
- [ ] cli/ - 大幅简化 (只保留基础入口)
- [ ] server.py (API服务) - 删除 (用 Mini-LLM-Server 替代)
- [ ] config.py - 简化 (只保留核心配置项)

---

## 删减说明 (Design Decisions)

### 原始模块 vs Mini 版本对照表

| 原始模块 | 状态 | 原因 / 替代方案 |
|----------|------|------------------|
| **adapter/** | ✅ 保留 | 核心组件，转换 Trace 格式 |
| **algorithm/** | ⚠️ 部分保留 | 保留 `base.py`, `fast.py`, `apo/`, 删除 `verl/`，新增 `mini_grpo.py` |
| **cli/** | ❌ 删除 | 复杂度高，用简单的训练脚本替代 |
| **config.py** | ⚠️ 简化 | 只保留核心配置项 |
| **emitter/** | ✅ 新增 | 原始 mini 版本缺失，需要从原始项目复制简化版 |
| **execution/** | ✅ 保留 | 核心组件，Runner 执行策略 |
| **instrumentation/** | ❌ 删除 | 复杂 monkey-patch，用 LLMProxy + Server 端 Trace 替代 |
| **litagent/** | ✅ 保留 | 核心组件，Agent 基类 |
| **llm_proxy.py** | ⚠️ 简化 | 保留核心代理功能，删除复杂 vllm 集成，**新增 LoRA 支持** |
| **logging.py** | ✅ 保留 | 日志基础设施 |
| **reward.py** | ✅ 保留 | 奖励计算基础 |
| **runner/** | ✅ 保留 | 核心组件，执行 Agent |
| **semconv.py** | ✅ 保留 | 语义约定，Trace 标准 |
| **server.py** | ❌ 删除 | 复杂 API 服务，用 Mini-LLM-Server 替代 |
| **store/** | ✅ 保留 | 核心组件，数据存储 |
| **tracer/** | ✅ 保留 | 核心组件，轨迹记录 |
| **trainer/** | ✅ 保留 | 核心组件，训练编排 |
| **types/** | ✅ 保留 | 核心类型定义，**包含 Hook 基类 (必须保留)** |
| **runner/** | ✅ 保留 | 核心组件，执行 Agent，**调用 Hook 生命周期** |

---

### 5. Hooks (生命周期回调)
- **原始**: `types/core.py` 定义了 `Hook` 基类，包含了 `on_trace_start`, `on_rollout_start`, `on_rollout_end`, `on_trace_end` 等回调。
- **状态**: **✅ 完全保留**
- **理由**: 
  - Hook 是 Trainer/Runner 串联流程的核心机制
  - `on_rollout_end` 是计算 Reward 的标准位置 (GRPO 必需)
  - **删除 Hook 需要重写 Trainer/Runner，得不偿失**
- **简化**: 只保留 `Hook` 基类接口 + 一个简单的 `LoggingHook`，删除依赖 wandb/mlflow 的复杂实现
| **utils/** | ⚠️ 简化 | 保留基础工具，删除复杂依赖 |
| **verl/** | ❌ 删除 | 用自定义 Mini-GRPO 替代 |

### 关键设计决策

1. **Trace 机制**:
   - 原始: `instrumentation/` (Monkey Patch openai/langchain 等)
   - Mini: `LLMProxy` (显式包裹) + `Mini-LLM-Server` (Server 端记录)
   - 理由: 避免脆弱的 Monkey Patch，更稳定

2. **RL 算法**:
   - 原始: `verl/` (需要 vllm + 大量显存)
   - Mini: `Mini-GRPO` (纯 PyTorch 实现)
   - 理由: 适配 4GB 显存环境

3. **LLM 推理**:
   - 原始: `vllm` (高吞吐，但显存占用大)
   - Mini: `Mini-LLM-Server` (FastAPI + Transformers + LoRA)
   - 理由: 显存可控，支持 LoRA 量化

4. **显存优化 (关键设计)**:
   - **方案**: LoRA (Low-Rank Adaptation)
   - **原理**: 只训练少量适配器参数 (1-2%)，冻结原始模型权重
   - **实测效果**: Qwen2-0.5B (498M参数) → 只训练 4.4M 参数 (0.88%)
   - **显存需求**: 从 4GB+ 降到 ~2GB
   - **实现**: 使用 `peft` 库 + LoraConfig
   - **可选进阶**: 4bit/8bit 量化 (bitsandbytes)

4. **Agent 无感**:
   - 原始: 需要 `import instrumentation` + Monkey Patch
   - Mini: Agent 代码只需改 `base_url` 指向 Mini-LLM-Server
   - 理由: 更简洁的集成方式

---

## Execution Strategy

### 5 Waves 执行计划

```
Wave 1 (立即开始 — 修复核心 import):
├── Task 1: 修复 adapter 模块 import
├── Task 2: 修复 algorithm 模块 import
├── Task 3: 修复 execution 模块 import
├── Task 4: 修复 litagent 模块 import
├── Task 5: 修复 runner 模块 import
├── Task 6: 修复 store 模块 import
├── Task 7: 修复 tracer 模块 import
├── Task 8: 修复 trainer 模块 import
├── Task 9: 修复 types/utils 模块 import
└── Task 10: 修复 __init__.py 导出

Wave 2 (Wave 1 后 — 添加缺失模块):
├── Task 11: 创建 config.py 配置模块
├── Task 12: 创建 emitter/__init__.py
├── Task 13: 创建 emitter/annotation.py
├── Task 14: 创建 emitter/reward.py
└── Task 15: 创建 emitter/message.py

Wave 3 (Wave 2 后 — 实现 Mini-LLM-Server):
├── Task 16: 创建 mini_llm_server.py (FastAPI)
├── Task 17: 集成 transformers/llama-cpp 后端
├── Task 18: 实现 /v1/chat/completions 接口
└── Task 19: 验证 OpenAI SDK 调用兼容性

Wave 4 (Wave 3 后 — 实现 Mini-GRPO + LoRA):

- [ ] 20. 创建 mini_grpo.py

  **What to do**:
  - 创建 `mini_agent_lighting/algorithm/mini_grpo.py`
  - 继承 `Algorithm` 类
  - **集成 LoRA 支持 (使用 peft 库)**

  **显存优化设计**:
  ```
  LoRA Config:
  - r=8 (rank)
  - lora_alpha=16
  - lora_dropout=0.05
  - target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  
  实测效果:
  - Qwen2-0.5B: 498M params → 4.4M trainable (0.88%)
  - 显存: ~4GB → ~2GB
  ```

  **Acceptance Criteria**:
  - [ ] 类结构正确
  - [ ] LoRA 可启用/禁用

- [ ] 21. 实现 GRPO Loss

  **What to do**:
  - 实现 advantage 计算
  - 实现 loss 函数
  - 只更新 LoRA 参数，冻结主模型

  **Acceptance Criteria**:
  - [ ] 单元测试通过

- [ ] 22. 实现参数更新

  **What to do**:
  - 集成 PyTorch Optimizer
  - **只优化 LoRA 参数**

  **Acceptance Criteria**:
  - [ ] 参数可更新

- [ ] 23. 删除 verl 依赖

  **What to do**:
  - 删除 `algorithm/verl/` 目录

  **Acceptance Criteria**:
  - [ ] 无 verl 引用

### Wave 5: 训练验证 (集成 Agent-Lightning 框架)

- [ ] 24. 集成 Agent-Lightning Trainer

  **What to do**:
  - 使用 `mini_agent_lighting.Trainer`
  - 配置 `MiniGRPO` 作为 algorithm
  - **确保 LoRA 参数被正确训练**

  **Acceptance Criteria**:
  - [ ] Trainer 可启动

- [ ] 25. 配置 Hooks 奖励计算

  **What to do**:
  - 使用 `on_rollout_end` Hook 计算奖励
  - 集成 reward 函数

  **Acceptance Criteria**:
  - [ ] Hook 被正确调用

- [ ] 26. 运行训练

  **QA Scenarios**:
  ```
  Scenario: 完整训练 (LoRA 模式)
    Tool: Bash
    Steps:
      1. python examples/run_grpo.py (已测试通过)
    Expected Result: Loss 下降，显存 < 4GB

  Scenario: Agent-Lightning 集成
    Tool: Bash  
    Steps:
      1. 使用 Trainer + MiniGRPO + Hooks
    Expected Result: 框架正常工作
  ```

- [ ] 27. 编写文档

  **What to do**:
  - 记录架构图 (Mini-LLM-Server + Mini-GRPO + LoRA)
  - 记录显存优化设计

  **Acceptance Criteria**:
  - [ ] 文档清晰

---

## Verification Commands
```bash
# 1. Import 测试
python -c "from mini_agent_lighting import Trainer"

# 2. LoRA + GRPO 测试 (已验证通过)
python examples/run_grpo.py
# 预期: trainable params: 4,399,104 || all params: 498,431,872 || trainable%: 0.8826
# 结果: 10 iterations 训练，正确率从 0/4 提升到 4/4

# 3. Agent-Lightning 集成测试
python examples/train_with_trainer.py
# 结果: 5 iterations 训练完成，集成正常工作

# 4. Server 测试
python -m mini_agent_lighting.utils.mini_llm_server &
curl http://localhost:8000/v1/chat/completions ...
```

---

## Wave 6: 代码清理 + 正确的使用示例

### 执行步骤 (已完成)

- [x] 6.1 删除 examples/run_grpo.py
- [x] 6.2 删除 examples/train_with_trainer.py
- [x] 6.3 删除 examples/train_mini.py
- [x] 6.4 保留 examples/benchmark_gsm8k.py
- [x] 6.5 创建 examples/math_agent_demo.py
- [x] 6.6 验证示例可运行

---

## Wave 7: 完整训练集成 (正确的 Agent-Lighting 方式)

### 问题
当前 benchmark_gsm8k.py 不合理: benchmark 里不应该包含训练

### MiniGRPO 实现选择

| 方案 | 优点 | 缺点 |
|------|------|------|
| **手搓** | 完全可控,无额外依赖 | 需要自己维护 |
| **trl** | 开源稳定,SGRPOTrainer 轻量 | 多一个依赖 |

**选择**: 支持两种,可切换

```python
# 方式 1: 手搓 MiniGRPO (当前实现)
from mini_agent_lighting.algorithm.mini_grpo import MiniGRPO
grpo = MiniGRPO(model, tokenizer)

# 方式 2: trl SGRPOTrainer
from trl import SGRPOTrainer
grpo = SGRPOTrainer(model=model, reward_function=compute_reward)
```

### 正确的 Agent-Lighting 使用方式

```
1. 定义 Agent (LitAgent 子类) - 负责推理
2. 定义 Hooks - 负责 reward 计算
3. 定义 Algorithm (MiniGRPO) - 负责优化
4. Trainer.fit() - 编排整个训练流程
5. 单独 benchmark 评估 - 不包含训练
```

### examples/ 目录结构 (正确)

```
examples/
├── train_math_agent.py  # 训练: Trainer + MathAgent + Hooks + MiniGRPO
└── benchmark.py         # 评估: 只评估模型,不训练
```

### 执行步骤

- [x] 7.1 删除 benchmark_gsm8k.py (不合理: benchmark 里跑训练)
- [x] 7.2 创建 examples/train_math_agent.py (正确的训练方式)
- [x] 7.3 创建 examples/benchmark.py (只评估,不训练)
- [ ] 7.4 使用 CoT prompt
- [ ] 7.5 训练 50 iterations
- [ ] 7.6 运行 benchmark 评估

---

## Wave 8 (未来): Legacy 代码清理

### 问题
当前 legacy 文件被主代码使用,不能直接删除

### Legacy 文件

| 文件 | 使用情况 | 清理方案 |
|------|---------|---------|
| `trainer/legacy.py` | `Trainer` 继承 `TrainerLegacy` | 重构 Trainer 移除继承 |
| `runner/legacy.py` | 导出 `LegacyAgentRunner` | 重构 Runner 移除导出 |

### 执行步骤 (未来)

- [ ] 8.1 重构 Trainer 不再继承 TrainerLegacy
- [ ] 8.2 重构 Runner 移除 LegacyAgentRunner
- [ ] 8.3 删除 trainer/legacy.py
- [ ] 8.4 删除 runner/legacy.py
- [ ] 8.5 验证功能正常

---

## 当前 examples/ 目录结构

```
examples/
├── benchmark_gsm8k.py    # ❌ 不合理: benchmark 里跑训练
└── math_agent_demo.py   # Demo,非完整训练
```

### 正确的使用方式

#### 1. 自定义 Agent (继承 LitAgent)

```python
from mini_agent_lighting.litagent import LitAgent
from mini_agent_lighting.types import Rollout, Task

class MathAgent(LitAgent):
    """数学问题求解 Agent"""
    
    def __init__(self, llm_client, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = llm_client
    
    def rollout(self, task: Task) -> Rollout:
        """执行单次推理"""
        prompt = self._build_prompt(task.input)
        response = self.llm_client.chat(prompt)
        return Rollout(
            input=task.input,
            output=response,
            traces=[...]
        )
```

#### 2. 使用 Hooks 计算 Reward

```python
from mini_agent_lighting.types import Hook

class GRPORewardHook(Hook):
    """在 rollout 结束后计算 reward"""
    
    def on_rollout_end(self, rollout: Rollout, task: Task):
        reward = self._compute_reward(rollout.output, task)
        rollout.metadata["reward"] = reward
        return rollout
```

#### 3. 使用 Trainer 编排

```python
from mini_agent_lighting import Trainer

trainer = Trainer(
    algorithm=MiniGRPO(model, tokenizer),
    agent=MathAgent(llm_client),
    hooks=[GRPORewardHook()],
    store=InMemoryLightningStore(),
    n_runners=4,
)
trainer.fit(dataset)
```

---

如果你发现模型的效果迟迟提不上
