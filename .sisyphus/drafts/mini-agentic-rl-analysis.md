# Draft: mini-agentic-rl 项目分析

## 目标
实现一个 mini 版的 agent-lighting，最大限度保留核心功能和代码结构

## 当前状态

### 原始代码 (agent-lightning)
- 位置: `/home/lianzhanbiao/workspace/agent-lightning/`
- 完整模块: adapter, algorithm, cli, config, emitter, execution, instrumentation, litagent, llm_proxy, logging, runner, store, tracer, trainer, types, utils, verl, server

### mini 版本 (mini_agent_lightning)
- 位置: `/home/lianzhanbiao/workspace/mini-agentic-rl/mini_agent_lightning/`
- 已有模块: adapter, algorithm, client, env_var, execution, litagent, llm_proxy, logging, reward, runner, semconv, store, tracer, trainer, types, utils

### 缺失模块 (mini版本)
- config.py - 配置管理
- emitter/ - 事件发射
- cli/ - 命令行入口
- server.py - API服务
- instrumentation/ - 插桩模块
- verl/ - verl集成

## 核心问题

**Import 错误**: mini 版本代码全部引用 `from agentlightning.xxx`，但包名应该是 `mini_agent_lightning`

```
ModuleNotFoundError: No module named 'agentlightning'
```

这是因为代码是从原始项目复制过来的，但忘记修改 import 语句。

## 技术决策

### 运行目标
- 最小目标: 能成功 import mini_agent_lightning 模块
- 中级目标: 能运行训练流程
- 完整目标: 实现完整 CLI + 服务

### 需要决策
- [ ] import 替换策略：全部替换为 mini_agent_lightning
- [ ] 缺失模块处理：全部删除引用还是创建 stub
- [ ] 是否需要完整 CLI
- [ ] 是否需要 server.py
