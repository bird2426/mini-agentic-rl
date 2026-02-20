# GSM8K Benchmark + 改进训练计划

## 目标
在 GSM8K 数据集上评估训练后的模型，与 SOTA 对比

## 当前状态

### 已完成
- ✅ Mini-GRPO + LoRA 训练框架
- ✅ 训练脚本 (examples/run_grpo.py)
- ✅ Benchmark 脚本 (examples/benchmark_gsm8k.py)
- ✅ 基础测试: 4GB VRAM 运行成功

### 待完成
- ⏳ 改进训练 (使用 CoT prompt + 更多数据)
- ⏳ 运行完整 GSM8K benchmark
- ⏳ 与 SOTA 对比分析

---

## 1. 问题分析

### SOTA 参考数据

| 模型 | GSM8K | 条件 |
|------|-------|------|
| Qwen2.5-0.5B | **36.92%** | 8-shot CoT |
| Llama-3.2-1B | 5.69% | 8-shot CoT |
| SmolLM2-360M | 4.55% | 8-shot CoT |

### 我们当前结果

| 测试 | 结果 | 说明 |
|------|------|------|
| 简单 prompt (无训练) | 15% | 无 CoT |
| 简单 prompt (训练后) | ~20-40% | 4 个问题训练 |
| **与 SOTA 差距** | **~20%** | prompt + 数据量 |

### 核心差距

1. **Prompt 差距**: 使用 CoT (Chain-of-Thought) vs 简单 prompt
2. **数据量差距**: 4 个问题 vs 8-shot CoT 示例

---

## 2. 改进方案

### 方案 A: CoT Prompt + 小数据集 (快速验证)

```python
COT_PROMPT = """Solve the following math problem step by step.

Example 1:
Problem: There are 5 birds on a tree. 2 fly away. How many birds are left?
Solution: 5 - 2 = 3. There are 3 birds left. #### 3

Example 2:
Problem: Tom has 10 apples. He gives 3 to Jerry. How many apples does Tom have now?
Solution: 10 - 3 = 7. Tom has 7 apples now. #### 7

Now solve this problem:
Problem: {problem}

Solution:"""
```

- 训练数据: 50-100 个 GSM8K train 问题
- 预期提升: 15% → 30%

### 方案 B: 完整 GRPO 训练 (长期)

- 训练数据: 1000+ GSM8K train 问题
- 使用 reward model 评估
- 预期提升: 30% → 40%

---

## 3. 执行计划

### Task 1: CoT Prompt 验证
- [ ] 1.1 更新 run_grpo.py 使用 CoT prompt
- [ ] 1.2 运行测试观察 baseline 变化

### Task 2: GSM8K 数据加载
- [ ] 2.1 修改脚本加载 GSM8K train split
- [ ] 2.2 提取问题和答案

### Task 3: 训练配置
- [ ] 3.1 设置 50 iterations
- [ ] 3.2 调整学习率 (1e-4 → 1e-5)
- [ ] 3.3 配置 LoRA 参数

### Task 4: 运行训练
- [ ] 4.1 训练 50 iterations
- [ ] 4.2 监控 Loss 变化
- [ ] 4.3 保存模型

### Task 5: Benchmark 测试
- [ ] 5.1 运行 GSM8K test (100 samples)
- [ ] 5.2 计算准确率
- [ ] 5.3 对比 SOTA

---

## 4. 预期结果

| 阶段 | 预期准确率 |
|------|-----------|
| Baseline (无训练, 简单 prompt) | 15% |
| Baseline (无训练, CoT prompt) | ~37% |
| 训练后 (CoT, 50 iters) | ~35-40% |

---

## 5. 风险与限制

1. **显存限制**: 4GB VRAM, 只能小 batch 训练
2. **数据限制**: 需要下载 GSM8K 数据集
3. **时间**: 50 iterations 可能需要 30-60 分钟
