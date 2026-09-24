# Planner-Reviewer System

这个文档描述了 sciMAS 的 **planner-reviewer 两阶段工作流生成系统**，它能够自动验证和修复 planner 生成的代码。

## 概述

在 `python-dsl` 模式下，sciMAS 现在使用两个 agent 协作生成和验证工作流代码：

1. **Planner Agent**: 生成初始的 Python DSL 工作流代码
2. **Reviewer Agent**: 验证代码的合法性，如果发现问题则自动修复

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     用户提交科学问题                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 1: Planner 生成工作流代码                               │
│  - 使用 planner.md prompt                                    │
│  - 生成 async def workflow(task) 代码                        │
│  - 可能包含格式错误（markdown、语法错误等）                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  验证循环（最多 3 次尝试）                                     │
│                                                             │
│  ┌──────────────────────────────────────────┐              │
│  │ 尝试验证代码                              │              │
│  │ parse_and_validate_workflow()            │              │
│  └──────────────────────────────────────────┘              │
│                  │                                          │
│                  ├─ 成功 ──→ 继续执行                        │
│                  │                                          │
│                  └─ 失败 ─┐                                 │
│                           ▼                                 │
│         ┌───────────────────────────────────┐              │
│         │ 阶段 2: Reviewer 修复代码          │              │
│         │ - 使用 planner-reviewer.md prompt │              │
│         │ - 分析错误信息                     │              │
│         │ - 修复代码                         │              │
│         │ - 返回修复后的代码                 │              │
│         └───────────────────────────────────┘              │
│                           │                                 │
│                           └──→ 重新验证                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  执行工作流（带错误恢复）                                      │
│                                                             │
│  如果执行时出现格式错误：                                      │
│  1. 捕获执行异常                                             │
│  2. 调用 Reviewer 修复代码                                   │
│  3. 重新验证并执行（最多 2 次重试）                            │
└─────────────────────────────────────────────────────────────┘
```

## 功能特性

### 1. 自动验证和修复

Reviewer agent 可以自动修复以下常见问题：

- **Markdown 包裹**: 移除 ```python 和 ``` 标记
- **JSON 输出**: 将 JSON 格式转换为 Python 代码
- **导入语句**: 移除不允许的 import
- **无效的 agent 调用**: 修复为只使用关键字参数
- **无效的角色名称**: 替换为合法的角色名
- **while 循环**: 转换为 for 循环
- **try/except 块**: 移除并简化
- **辅助函数**: 内联到 workflow 函数中
- **缺少 return**: 添加显式返回语句
- **语法错误**: 修复 Python 语法问题

### 2. 多次重试机制

- **验证阶段**: 最多 3 次尝试修复验证错误
- **执行阶段**: 最多 2 次尝试修复执行错误
- 每次尝试都会通过事件系统记录，便于调试

### 3. 事件追踪

系统会发出以下事件用于监控：

- `workflow_validation_failed`: 验证失败
- `reviewer_found_issues`: Reviewer 发现问题
- `workflow_fixed_by_reviewer`: Reviewer 修复了代码
- `workflow_execution_failed`: 执行失败
- `workflow_fixed_after_execution_error`: 执行错误后修复

### 4. 置信度评分

Reviewer 会返回置信度分数 (0.0-1.0)，表明修复的可靠性：
- `>= 0.8`: 高置信度，修复应该有效
- `0.5-0.8`: 中等置信度，可能需要再次审查
- `< 0.5`: 低置信度，建议人工介入

## 使用方法

### 通过 Web Dashboard

1. 启动 dashboard:
   ```bash
   python web_dashboard.py --port 7860
   ```

2. 在配置中选择:
   - **Planner mode**: `python-dsl`
   
3. 提交问题后，系统会自动：
   - 生成工作流代码
   - 验证并修复（如需要）
   - 执行工作流

### 通过代码

```python
from orchestrator import SciMASOrchestrator

# 创建 orchestrator，指定 python-dsl 模式
orchestrator = SciMASOrchestrator(planner_mode="python_dsl")

# 提交问题，系统自动处理验证和修复
problem = "Calculate the energy of a photon with wavelength 500 nm."
report = orchestrator.run(problem, roles=["physicist", "generalist"])

print(report.final_answer)
```

## Reviewer Prompt

Reviewer 使用专门的 prompt (`prompts/planner-reviewer.md`)，它定义了：

1. **输入格式**: JSON context bundle，包含：
   - `workflow_source`: 待审查的代码
   - `validation_error`: 验证错误信息（可选）
   - `execution_error`: 执行错误信息（可选）
   - `attempt_number`: 尝试次数

2. **输出格式**: JSON 响应，包含：
   - `is_valid`: 代码是否有效
   - `issues`: 发现的问题列表
   - `fixed_source`: 修复后的代码
   - `changes_made`: 做出的修改
   - `confidence`: 置信度分数

3. **修复规则**: 详细的 DSL 限制和常见问题修复指南

## 测试

运行测试套件：

```bash
# 单元测试
python -m pytest tests/test_planner_reviewer.py -v

# 演示脚本
python demo_reviewer.py
```

## 限制

1. **最大尝试次数**: 
   - 验证: 3 次
   - 执行: 2 次
   - 达到上限后会抛出异常

2. **Reviewer 能力**:
   - 只能修复格式和语法问题
   - 不能修复逻辑错误
   - 不能生成全新的工作流

3. **成本**:
   - 每次修复需要额外的 LLM 调用
   - 建议监控 token 使用情况

## 调试

如果遇到问题：

1. **查看事件日志**: 在 web dashboard 的 Event log 面板中
2. **检查输出目录**: `runs-dashboard/<timestamp>/` 中的：
   - `report.json`: 完整的执行记录
   - `trace.jsonl`: 事件流
   - `report.md`: 人类可读的报告

3. **启用详细日志**:
   ```python
   orchestrator = SciMASOrchestrator(
       planner_mode="python_dsl",
       event_callback=lambda e: print(f"Event: {e}")
   )
   ```

## 未来改进

- [ ] 支持更复杂的代码转换
- [ ] 添加代码质量评分
- [ ] 支持增量修复（只修改有问题的部分）
- [ ] 缓存常见修复模式
- [ ] 与 IDE 集成，提供实时反馈

## 相关文件

- `orchestrator.py`: 主要实现
- `prompts/planner-reviewer.md`: Reviewer agent prompt
- `prompts/planner.md`: Planner agent prompt
- `workflow_dsl.py`: DSL 验证器
- `tests/test_planner_reviewer.py`: 测试套件
- `demo_reviewer.py`: 演示脚本
