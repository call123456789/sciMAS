# 快速开始指南：Planner-Reviewer 系统

## 概述

sciMAS 现在包含一个两阶段的代码验证系统，可以自动修复 planner 生成的工作流代码中的格式错误。

## 架构

```
用户问题
   ↓
Planner Agent → 生成 Python 代码
   ↓
验证代码
   ↓
失败? → Reviewer Agent → 修复代码 → 重新验证 (最多3次)
   ↓
成功? → 执行工作流
   ↓
执行失败? → Reviewer Agent → 修复代码 → 重新执行 (最多2次)
   ↓
返回结果
```

## 主要文件

### 新增的文件

1. **`prompts/planner-reviewer.md`** - Reviewer agent 的 prompt
   - 定义验证规则
   - 说明常见错误和修复方法
   - 指定输入/输出格式

2. **`tests/test_planner_reviewer.py`** - 单元测试
   - 测试各种错误场景
   - 测试修复功能
   - 包含 mock 示例

3. **`demo_reviewer.py`** - 演示脚本
   - 可以直接运行查看效果
   - 展示3种常见错误的修复

4. **`PLANNER_REVIEWER.md`** - 完整文档
   - 详细的架构说明
   - 使用指南
   - 调试建议

5. **`IMPLEMENTATION_SUMMARY.md`** - 实现总结
   - 技术细节
   - 代码修改说明
   - 验证方法

### 修改的文件

1. **`orchestrator.py`**
   - 新增 `_review_and_fix_workflow()` 方法
   - 修改 `_plan_python_dsl()` 添加验证循环
   - 修改 `_run_python_dsl_workflow()` 添加执行错误恢复

2. **`README.md`**
   - 添加 Python DSL 模式说明
   - 链接到详细文档

## 快速测试

### 1. 运行演示脚本

```bash
cd /Users/a123/Documents/games/sciMAS
python demo_reviewer.py
```

这会展示：
- Markdown 包裹代码的修复
- 无效 agent 调用的修复
- 缺失 return 语句的修复

### 2. 运行单元测试

```bash
cd /Users/a123/Documents/games/sciMAS
python -m pytest tests/test_planner_reviewer.py -v
```

### 3. 使用 Web Dashboard

```bash
cd /Users/a123/Documents/games/sciMAS
python web_dashboard.py --port 7860
```

然后：
1. 在浏览器中打开 http://localhost:7860
2. 在 "Planner mode" 中选择 "Python DSL"
3. 提交一个科学问题
4. 在 "Event log" 中查看修复过程

## Reviewer 可以修复的问题

1. ✓ Markdown 包裹 (```python ... ```)
2. ✓ JSON 格式输出
3. ✓ Import 语句
4. ✓ 无效的 agent 调用（位置参数）
5. ✓ 无效的角色名称
6. ✓ While 循环
7. ✓ Try/except 块
8. ✓ 辅助函数
9. ✓ 缺少 return 语句
10. ✓ 语法错误

## 事件追踪

系统会发出以下事件，可以在 dashboard 的 Event log 中看到：

- `workflow_validation_failed` - 验证失败
- `reviewer_found_issues` - 发现问题
- `workflow_fixed_by_reviewer` - 代码已修复
- `workflow_execution_failed` - 执行失败
- `workflow_fixed_after_execution_error` - 执行错误后修复

## 重试限制

- **验证阶段**: 最多 3 次尝试
- **执行阶段**: 最多 2 次重试
- 达到限制后会抛出异常

## 常见问题

**Q: Reviewer 调用会增加成本吗？**
A: 是的，每次修复需要一次额外的 LLM 调用。但这比人工修复更快更便宜。

**Q: 如何查看修复了什么？**
A: 查看 Event log 或 `runs-dashboard/<timestamp>/trace.jsonl` 文件。

**Q: Reviewer 可以修复逻辑错误吗？**
A: 不可以，Reviewer 只能修复格式和语法问题，不能修复业务逻辑。

**Q: 如何禁用 Reviewer？**
A: 目前无法禁用，但可以在 `orchestrator.py` 中将 `MAX_REVIEW_ATTEMPTS` 设为 1。

## 下一步

- 阅读 `PLANNER_REVIEWER.md` 了解详细架构
- 阅读 `IMPLEMENTATION_SUMMARY.md` 了解实现细节
- 运行测试验证功能
- 在实际问题上测试系统

## 需要帮助？

查看以下文件：
- `PLANNER_REVIEWER.md` - 完整文档
- `prompts/planner-reviewer.md` - Reviewer prompt
- `tests/test_planner_reviewer.py` - 测试示例
- `demo_reviewer.py` - 演示代码
