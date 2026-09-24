# Planner-Reviewer 实现总结

## 任务目标

实现一个两阶段的代码验证和修复系统：
1. **Planner** 生成 MAS 工作流代码
2. **Reviewer** 验证代码合法性，如果不合法则修复
3. 如果执行时因格式错误失败，Reviewer 再次修复并重试

## 已完成的工作

### 1. 创建 Reviewer Agent Prompt
**文件**: `prompts/planner-reviewer.md`

- 定义了 Reviewer 的职责：验证、识别问题、修复代码
- 明确了 DSL 限制规则（只允许特定的 Python 语法）
- 指定了输入/输出格式（JSON）
- 列举了 10+ 种常见问题和修复方法
- 包含详细的修复指南和置信度评分

### 2. 修改 Orchestrator 实现验证循环

**文件**: `orchestrator.py`

#### 新增方法：

##### `_review_and_fix_workflow()`
```python
def _review_and_fix_workflow(
    workflow_source: str,
    validation_error: Optional[str] = None,
    execution_error: Optional[str] = None,
    attempt_number: int = 1,
    problem: str = "",
) -> dict[str, Any]
```
- 调用 reviewer agent 验证和修复代码
- 接收错误信息和尝试次数
- 返回修复结果（包含 is_valid, issues, fixed_source, changes_made, confidence）

#### 修改的方法：

##### `_plan_python_dsl()`
- 添加了验证循环（最多 3 次尝试）
- 在 planner 生成代码后立即验证
- 如果验证失败，调用 reviewer 修复
- 发出事件追踪每次尝试
- 重复验证直到成功或达到最大尝试次数

主要流程：
```
planner 生成 → 验证 → 失败 → reviewer 修复 → 重新验证 → ...
```

##### `_run_python_dsl_workflow()`
- 添加了执行错误恢复机制（最多 2 次重试）
- 捕获执行异常
- 调用 reviewer 修复执行错误
- 重新验证并执行修复后的代码
- 发出执行错误和修复事件

主要流程：
```
执行 → 异常 → reviewer 修复 → 重新验证 → 重新执行 → ...
```

### 3. 事件系统集成

新增的事件类型：
- `workflow_validation_failed`: 验证失败时触发
- `reviewer_found_issues`: Reviewer 发现问题
- `workflow_fixed_by_reviewer`: Reviewer 修复了代码
- `workflow_execution_failed`: 执行失败时触发
- `workflow_fixed_after_execution_error`: 执行错误后修复

这些事件会：
- 在 web dashboard 的 Event log 中显示
- 记录到 `trace.jsonl` 文件中
- 可用于监控和调试

### 4. 测试套件

**文件**: `tests/test_planner_reviewer.py`

包含的测试：
1. `test_reviewer_fixes_markdown_wrapped_code()` - 测试移除 markdown 标记
2. `test_reviewer_fixes_missing_return()` - 测试添加缺失的 return
3. `test_reviewer_fixes_invalid_agent_call()` - 测试修复无效的函数调用
4. `test_plan_with_reviewer_loop()` - 测试完整的规划流程
5. `test_reviewer_gives_up_after_max_attempts()` - 测试最大尝试次数限制

### 5. 演示脚本

**文件**: `demo_reviewer.py`

演示功能：
- 展示 3 种常见错误和修复过程
- 可直接运行查看效果
- 包含详细的注释和输出

### 6. 文档

**文件**: `PLANNER_REVIEWER.md`

完整的中文文档，包含：
- 系统架构图
- 功能特性说明
- 使用方法（Web Dashboard 和代码）
- Reviewer prompt 解释
- 测试指南
- 调试建议
- 未来改进计划

**更新**: `README.md`
- 添加了 Python DSL 模式的工作流说明
- 链接到详细文档

## 技术细节

### 重试策略

**验证阶段**（在 `_plan_python_dsl` 中）:
```python
MAX_REVIEW_ATTEMPTS = 3
attempt = 1

while attempt <= MAX_REVIEW_ATTEMPTS:
    try:
        validate_workflow(code)
        break  # 成功
    except ValidationError:
        if attempt >= MAX_REVIEW_ATTEMPTS:
            raise  # 放弃
        
        # 调用 reviewer 修复
        fixed = review_and_fix(code)
        code = fixed
        attempt += 1
```

**执行阶段**（在 `_run_python_dsl_workflow` 中）:
```python
MAX_EXECUTION_RETRIES = 2

for attempt in range(1, MAX_EXECUTION_RETRIES + 1):
    try:
        result = execute_workflow(code)
        break  # 成功
    except ExecutionError as e:
        if attempt >= MAX_EXECUTION_RETRIES:
            raise  # 放弃
        
        # 调用 reviewer 修复
        fixed = review_and_fix(code, execution_error=str(e))
        code = fixed
        # 重新验证
        validate_workflow(code)
```

### 数据流

**Reviewer 输入** (JSON):
```json
{
  "workflow_source": "待审查的 Python 代码",
  "validation_error": "验证错误信息（可选）",
  "execution_error": "执行错误信息（可选）",
  "attempt_number": 1,
  "problem_summary": "问题摘要"
}
```

**Reviewer 输出** (JSON):
```json
{
  "is_valid": true/false,
  "issues": ["问题1", "问题2"],
  "fixed_source": "修复后的代码",
  "changes_made": ["修改1", "修改2"],
  "confidence": 0.95
}
```

### 错误处理

1. **Reviewer 返回无效 JSON**: 返回默认响应，标记为无效
2. **Reviewer 未能修复**: 抛出 RuntimeError，包含详细错误信息
3. **达到最大尝试次数**: 抛出 RuntimeError，包含最后的错误和代码
4. **修复后的代码仍然无效**: 抛出 RuntimeError，继续尝试下一轮

## 与现有系统的集成

### Web Dashboard
- 不需要修改前端代码
- 事件会自动显示在 Event log 中
- 修复过程对用户透明

### 命令行
- 不需要修改调用方式
- 只需确保 `planner_mode="python_dsl"`

### 测试框架
- 可以通过 mock runner 测试修复逻辑
- 不需要实际的 Claude API 调用

## 验证

可以通过以下方式验证实现：

1. **运行演示脚本**:
   ```bash
   python demo_reviewer.py
   ```

2. **运行单元测试**:
   ```bash
   python -m pytest tests/test_planner_reviewer.py -v
   ```

3. **使用 Web Dashboard**:
   ```bash
   python web_dashboard.py --port 7860
   # 选择 python-dsl 模式运行问题
   ```

## 优势

1. **自动修复**: 减少了人工干预的需要
2. **透明度**: 通过事件系统可以看到每一步修复过程
3. **鲁棒性**: 多次重试提高了成功率
4. **可调试**: 详细的错误信息和修复历史
5. **可扩展**: 可以轻松添加新的修复规则

## 限制

1. **成本**: 每次修复需要额外的 LLM 调用
2. **时间**: 多次重试会增加总体执行时间
3. **能力边界**: Reviewer 只能修复格式问题，不能解决逻辑错误
4. **最大尝试次数**: 硬编码的限制可能不适合所有情况

## 未来改进

1. **智能重试**: 根据错误类型动态调整重试次数
2. **缓存**: 缓存常见的修复模式
3. **增量修复**: 只修改有问题的部分而不是整个代码
4. **质量评分**: 对生成的代码进行质量评估
5. **可配置限制**: 允许用户配置最大尝试次数
6. **性能优化**: 批量验证多个候选修复
7. **学习机制**: 从过去的修复中学习

## 文件清单

新增文件：
- `prompts/planner-reviewer.md`
- `tests/test_planner_reviewer.py`
- `demo_reviewer.py`
- `PLANNER_REVIEWER.md`
- `IMPLEMENTATION_SUMMARY.md`

修改文件：
- `orchestrator.py` (添加 `_review_and_fix_workflow`, 修改 `_plan_python_dsl`, `_run_python_dsl_workflow`)
- `README.md` (添加 Planner-Reviewer 说明)

## 总结

实现了一个完整的两阶段代码生成和验证系统，能够：
- 自动检测和修复 planner 生成的代码格式错误
- 在执行时捕获错误并重新修复
- 通过事件系统提供完整的可观察性
- 包含完整的测试和文档

系统现在更加鲁棒，能够处理更多边界情况，减少了因代码格式问题导致的失败。
