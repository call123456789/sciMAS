# DrugSDA-Tool 集成指南

## 概述

本项目已成功集成 DrugSDA-Tool SCP Server，作为一个远程药物分析工具集。该工具提供了基于 RDKit、Open Babel 和 BioPython 的分子分析功能，包括：

- 分子式转换与计算
- 药物化学性质预测 (QED, logP, PSA, 等)
- 分子结构数据解析
- 结合口袋属性分析

## 集成方式

DrugSDA-Tool 通过本地 MCP 服务器包装器进行集成，该包装器在 `tools/pharma/drug_sda_server.py` 中实现，并在 `config/mcp.json` 中注册。`pharma-drug-sda` 服务作为一个 MCP 服务器启动，通过 streamable-http 协议连接到远程服务器。

## 环境变量配置

需要设置以下环境变量之一以提供 API 密钥：

- `DRUGSDA_API_KEY` (推荐)
- `SCP_HUB_API_KEY`

### 配置步骤

1. **获取 API 密钥**：从 [上海人工智能实验室 SCP-HUB](https://scp.intern-ai.org.cn) 获取访问密钥

2. **设置环境变量** (在运行脚本前)

   ```bash
   # 对于 macOS/Linux
   export DRUGSDA_API_KEY="your-api-key-here"
   
   # 对于 Windows (PowerShell)
   $env:DRUGSDA_API_KEY="your-api-key-here"
   ```

3. **可选：写入本机配置文件 `config/mcp.local.json`**

   `config/mcp.json` 会被提交，因此**不要**把密钥写进去。改为写入同目录下已被
   `.gitignore` 忽略的 `config/mcp.local.json`，启动时由
   `ClaudeRunner._resolve_mcp_config` 合并进被跟踪的配置：

   ```json
   {
     "mcpServers": {
       "pharma-drug-sda": {
         "env": { "DRUGSDA_API_KEY": "your-actual-api-key" }
       }
     }
   }
   ```

   只读取 `env` 字段，`command` / `args` 会被忽略，所以该文件无法把服务器
   指向别的脚本。优先级为环境变量 > 该文件，与 OpenAlex 密钥一致。

## 可用工具

集成后，以下角色可以使用 DrugSDA-Tool 的功能：

### `pharmacist` (药剂师)
- 完整访问所有药物发现、数据分析和 DrugSDA-Tool 功能

### `drug-discovery-scientist` (药物发现科学家)
- 访问 DrugSDA-Tool 和药物发现工具

### `pharma-data-specialist` (药物数据专家)
- 仅访问药物数据库查询工具

### `pharma-ml-engineer` (药物 ML 工程师)
- 仅访问 AutoML 训练工具

## 使用示例

### 在脚本中直接调用

可以通过以下方式使用：

```python
from sciMAS.orchestrator import SciMASOrchestrator

# 使用药物发现的专家角色
orchestrator = SciMASOrchestrator()
plan, planner = orchestrator.plan(
    problem="给出一组分子，计算它们的 QED 分数和药物化学性质",
    roles=["drug-discovery-scientist", "pharmacist"]
)
report = orchestrator.run(
    problem="SMILES: CCl3CCl3, C1=CC=CC=C1",
    roles=["drug-discovery-scientist"]
)
```

### 通过命令行

```bash
python run.py --problem "计算以下分子的 QED: CC(C)C1=CC=CC=C1, N[C@@H](Cc1ccc(O)cc1)C(=O)O" --roles drug-discovery-scientist
```

### 手动测试服务

可以独立运行包装器进行测试：

```bash
python tools/pharma/drug_sda_server.py
```

然后使用任何 MCP 客户端连接。

## 工具详情

### `calculate_mol_drug_chemistry`
计算分子的药物化学性质，返回以下指标：
- QED (药物可能性)
- logP (脂水分配系数)
- PSA (极性表面积)
- 分子量
- 氢键供/受体
- 旋转键数

**参数**:
- `smiles_list` (list[str]): SMILES 字符串列表

**返回**:
- JSON 对象，包含每个分子的完整药物化学分析

## 故障排除

### 连接失败
- 检查 API 密钥是否正确设置
- 验证网络连接是否能访问 `https://scp.intern-ai.org.cn`
- 尝试手动连接测试：`python -c "from tools.pharma.drug_sda_server import _proxy; import asyncio; asyncio.run(_proxy.connect())"`

### 工具调用失败
- 查看 `tools/pharma/drug_sda_server.py` 的日志输出
- 确认远程服务可用
- 检查输入 SMILES 格式是否有效

### 环境变量问题
- 确认 `DRUGSDA_API_KEY` 在运行环境中已设置
- 避免在 `config/mcp.json` 中硬编码密钥（应使用环境变量）

## 性能优化

- 包装器使用长连接以减少握手开销
- 自动重连机制保证调用稳定性
- 推荐使用 API 密钥以获得更好的服务质量

## 相关资源

- [DrugSDA-Tool 官方文档](https://scp.intern-ai.org.cn)
- [上海人工智能实验室 SCP-HUB](https://scp.intern-ai.org.cn)
- [RDKit 功能](https://www.rdkit.org)
