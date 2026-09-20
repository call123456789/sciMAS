# sciMAS 数据集分析报告

本文档对 `/Users/a123/Documents/games/sciMAS/dataset/` 下的六个本地数据集进行横向分析。所有数据都通过 `tests/dataset.py` 装载、`tests/grader.py` 评分（按 `Problem.dataset_name` 分派）。

---

## 一、数据集速览

| 数据集 | 题目量 | 格式 | 任务类型 | 与药物发现相关性 | 单题复杂度 | 评估器复杂度 |
|---|---|---|---|---|---|---|
| **BiomniEval1** | 1 个 parquet 表 | QA | 生物医学 QA 套件 | **~10–15%** | 极低 | 极低（本地确定性） |
| **SciPredict** | 405 | CSV | 论文实验结果预测 | **~20–25%** | 中 | 中（MCQ/数值/自由格式三种判定） |
| **ResearchClawBench** | HF 表 | 端到端研究复现 | 论文复现 | **~25–40%** | 极高 | 极高（多模态 LLM 评委） |
| **DrugDiscoveryBench** | 82 Harbor 任务 | 多步骤研究流水线 | 临床/药靶/筛选/分析 | **~90%** | 高 | 高（LLM-judge + 代理打分） |
| **MADD** | 100 | JSONL | 多靶点分子生成 | **100%** | 高 | 中–高（SMILES 抽取 + 工具覆盖） |
| **SMDDBench** | 502 | 502 个 `task.yaml` | 药物设计 5 类子任务 | **100%** | 高–极高 | 极高（外部 Docker 评估器） |

---

## 二、按三个维度排序

### 1. 与药物发现相关性（高 → 低）

**SMDDBench (100%) ≈ MADD (100%) > DrugDiscoveryBench (~90%) > ResearchClawBench (~25–40%) > SciPredict (~20–25%) > BiomniEval1 (~10–15%)**

- **SMDDBench / MADD**：100%。MADD 整张表都是疾病子任务（KRAS G12C、PCSK9、P-glycoprotein、Sclerosis、Parkinson、Drug_Resistance 等）；SMDDBench 的 5 个子类型（药效团识别、相互作用位点、Scaffold Hopping、Lead Optimization、Fragment Assembly）全部是分子层级的药物设计。
- **DrugDiscoveryBench**：82 个 Harbor 任务几乎全部围绕药靶、临床试验、FDA 药物筛选展开（CB2 激动剂筛选、Moertel 临床数据复算等）。
- **ResearchClawBench**：领域跨学科，仅约 1/3 是生物医药相关。
- **SciPredict**：405 题横跨生物/化学/物理三领域，仅约 25% 直接是药物发现（Analytical/Medicinal Chemistry 等实验结果预测）。
- **BiomniEval1**：偏临床 QA + 基因组学，`gwas_causal_gene_pharmaprojects` 子集是唯一与药物研发直接相关的子任务。

### 2. 单题复杂度（低 → 高）

**BiomniEval1 < SciPredict < MADD < DrugDiscoveryBench < SMDDBench < ResearchClawBench**

- **BiomniEval1**：单步查询/单选（A/B/C/D、OMIM ID、approved symbol）。
- **SciPredict**：读 paper + 实验设置 + 应用背景知识 + 推理一个实验结果（MCQ/数值/短答）。
- **MADD**：把题目拆成 1–4 个疾病子任务，分别调用 `gen_mols_<disease>` 工具，组合生成分子 + 性质预测表。
- **DrugDiscoveryBench**：多步骤研究流水线（加载多个 CSV、调用 biomni 工具箱、写结构化答案）。
- **SMDDBench**：解析 reference + complex.pdb、跑 Boltz-1 折叠、跑 8 项 ADMET 预测、满足 ~14 条硬约束 + 2–4 个优化目标 + 3–4 个 hold-constant 容忍度，输出一个 SMILES。
- **ResearchClawBench**：完整的研究论文复现——数据 + 相关工作 + 目标论文，产出含方法/数值结果/局限性的报告；目标论文与 checklist 在评测时才对选手开放。

### 3. 评估器复杂度（低 → 高）

**BiomniEval1 < SciPredict < MADD < ResearchClawBench < DrugDiscoveryBench < SMDDBench**

- **BiomniEval1**：本地确定性，按 `task_name` 分派——MCQ 字母大小写不敏感、JSON 字段比对、OMIM ID 字符串比对、token 重叠。**最易在本地复现**。
- **SciPredict**：三种确定性路径——MCQ 抽取 A-H 字母 + 集合相等、数值用 `89-99` 区间正则、free-form 用 rubrics.csv 的关键词加权。**完全本地化**。
- **MADD**：官方是 SSA × TS；本地是 `grade_madd_answer` + `grade_madd_tool_coverage`——正则提取 SMILES + 工具名映射（`gen_mols_lung_cancer` ↔ `case="lung cancer"`）。
- **ResearchClawBench**：官方多模态 LLM judge（agent 报告 vs. 目标论文 + checklist）；本地用 `grade_checklist_answer` 做关键词/文本加权（≥ 0.5 通过）。**官方与本地差异大**。
- **DrugDiscoveryBench**：官方 LLM judge 比对 `rubrics.json` 的 `ground_truth + outcome_rubrics + process_rubrics`；当前镜像的 rubrics 是空占位，本地只能跑 `avg_score × 100` 代理。**官方评估器依赖 gated 数据集**。
- **SMDDBench**：官方 Docker 评估器（RDKit + Boltz-1 + 多指标），sciMAS 本地**不会重新计算这些指标**，只能读 `result.json/score`；否则返回 0.0 并提示"请运行上游 Docker 评估器"。**最重**。

---

## 三、每个数据集的典型例子（2–3 个）

### 1. BiomniEval1

- `lab_001`（`lab_test_analyzing`）—— "Which lab result is most abnormal? A. ALT B. sodium"，标准答案 `"A"`。评估：MCQ 字母大小写不敏感。
- `gene_001`（`gene_name_conversion`）—— "Convert the alias p53 to the approved symbol."，答案 `"TP53"`。评估：标识符字符串相等。
- `patient_001`（`patient_gene_detection`）—— `answer: {"causal_gene": ["BRCA1"]}`。评估：JSON 集合相等。
- `gwas_causal_gene_pharmaprojects` 子集—— 本数据集里唯一直接药物研发相关的子任务。

### 2. SciPredict

- **MCQ（Physics / Warm Dense Matter）**：OMEGA 激光装置、60 路激光压 Cu 箔、VISAR/XAS 测量，问 shock 与 K-edge 的最可能行为，`CLEAN_GTA = "B"`。
- **数值（Analytical Chemistry）**：RapidPlate™ 电化学 AST vs. VITEK 2 / Kirby–Bauer，89 份尿样的整体 categorical agreement（%），`GTA = "94% [89% - 99%]"`、`CLEAN_GTA = "89-99"`，用区间正则判定。
- **自由格式（Neurobiology）**：VTA 多巴胺 → pBLA/aBLA 光遗传、恐惧条件化/消退/提取，rubrics 提供 5 条加权要点（消退是新学习、aBLA Rspo⁺ 负价、VTA DA 教学信号等）。

### 3. ResearchClawBench

- **任务形式**：每条 = `task_id, domain, task_tag, task` 描述 + `task_info_json`（含数据清单、`target_paper_path`、`checklist_json`），prompt 注入 Task ID / Domain / Task tag / Research objective / Data manifest / Related work papers，并明确"目标论文与 checklist 不允许在答题时使用，仅供评分"。
- **典型领域**：覆盖生物医药/化学/ML/物理，**含药物发现子任务但比例低**。
- **官方评估**：多模态 LLM judge；本地代理：基于 checklist 的关键词/文本加权（≥ 0.5 通过）。

### 4. DrugDiscoveryBench

- `69b025e20c10fe76b7aaf812`（结肠癌化疗临床统计）：基于 Moertel 1990 的 `survival::colon` 数据，比较"非经典化疗药比例"是否与历史趋势一致，要求 `YES/NO` + 两个百分比，预期输出如 `"NO, 30.0% vs 0.0%"`。
- `6a232796afa0e46251ad3d57`（CB2 激动剂筛选）：在 EvE bio 数据集中筛 FDA-approved、能激活 CB2（P34972）β-arrestin-2 招募、排除 `%Cell Death > 90%` 的化合物，输入 `dbfda-substances.csv` + `fda-substances.csv`，输出 5 列表（Drug/Result/CB2 Max/ZINC-DB-FDA?/ZINC-DSSTOX-FDA?）。

### 5. MADD

- `MADD_0001`（4 个疾病子任务）："Create 1 selective molecule that targets KRAS G12C and doesn't affect HRAS or NRAS" + 多发性硬化 + 血脂异常 + 帕金森混合设计；期望工具 `gen_mols_lung_cancer, gen_mols_multiple_sclerosis, gen_mols_dyslipidemia, gen_mols_parkinson`，输出分子表（Molecules | Docking | QED | SA | PAINS | SureChEMBL | Glaxo | Brenk | BBB | IC50）。
- `MADD_0005`（case = `Drug_Resistance, dyslipidemia`）："Design novel small molecule inhibitors targeting P-glycoprotein efflux pumps to reverse multidrug resistance. Generate a novel PCSK9 inhibitor…"
- `MADD_0050`（case = `Chat agent`）：直接对一个 SMILES 提问毒理学（"Does `<SMILES>` adversely affect the blood and lymphatic system?"），期望走 `make_answer_chat_model` 路径。

### 6. SMDDBench

- `smdd_001_O00748_0`（Cocaine esterase 药效团识别）—— 输入 `actives.smi / inactives.smi / protein.fasta`，写 `solution.py: check_pharmacophore(smiles) -> bool`；官方指标：actives Recall ≥ 0.8，inactives Specificity ≥ 0.8。
- `smdd_004_CDK2_1`（Lead Optimization 主体）—— 输入 `complex.pdb + reference.sdf + baseline_values`，目标：caco2 ≥ baseline + 0.3、bbb ≥ baseline + 0.1；hold-constant：binding_affinity 在 -2.6495 ± 0.3、ppb 在 82.7638 ± 5.0、ames 在 0.0719 ± 0.1；硬约束：MW<600、LogP -1..5、TPSA<140、HBD≤5、HBA≤10、rot bonds≤10、charge -2..+2、SA<4.5、无 PAINS/Brenk、Tanimoto ≥ 0.7、Boltz binding_probability_binary > 0.7；输出单个 SMILES 到 `solution.smi`。
- `smdd_005_P11362_2`（Fragment-to-Lead）—— 用 2 个片段（pyrrole + phenyl）+ `pocket.pdb` 设计单分子；额外要求片段 SMARTS 嵌入、片段链接、Boltz affinity_pred_value < -2.4162、片段 3D RMSD < 2.0 Å。

---

## 四、综合定位（一句话）

- **BiomniEval1** —— 入门级生物医学 QA 沙盒（轻量、确定、本地可全跑）。
- **SciPredict** —— 跨学科论文实验结果预测（含一定化学/药物子集）。
- **ResearchClawBench** —— 端到端论文复现（最难的"开放式研究"任务）。
- **DrugDiscoveryBench** —— 真实生物医药研究工作流（含临床/筛选/统计/分析）。
- **MADD** —— 多疾病子任务驱动的分子生成（100% 药物，结构化产出）。
- **SMDDBench** —— 最硬的分子层药物设计（5 个子类型，覆盖药效团 → 片段链接 Lead Optimization，依赖 Boltz-1 + Docker 评估器）。

---

## 五、关键引用文件

- `/Users/a123/Documents/games/sciMAS/tests/dataset.py` —— 装载器 & `Problem` 工厂方法（每个数据集一个）。
- `/Users/a123/Documents/games/sciMAS/tests/grader.py` —— 每个数据集的评分函数，在 `grade_problem` 中按 `problem.dataset_name` 分派。
- `/Users/a123/Documents/games/sciMAS/tests/runner.py` —— 端到端编排运行器，产出 `run.json` 与 `summary.md`，含各数据集得分行（`madd_fa_avg`、`ddb_score_100`、`scipredict_score_100`、`smdd_score_100`、`biomni_eval1_score_100`）。
- 各数据集的 `test_*.py` —— 断言装载行数、`dataset_name`/`subject`/`topic`、打分行为。

---

## 六、完备性审计（GT / 验证代码 / 外部依赖）

| 数据集 | GT 是否就绪 | 本地验证代码 | 外部依赖 | 端到端可跑 |
|---|---|---|---|---|
| **BiomniEval1** | **433/433**（100%） | ✅ 纯 Python stdlib | 无 | **✅ 完全可跑** |
| **DrugDiscoveryBench** | **0/82**（全部占位） | ⚠️ 仅代理打分 | Docker + gated HF + LLM judge + Harbor | **❌ 需要 `populate_rubrics.py`** |
| **MADD** | **100/100**（100%） | ✅ SMILES 子串 + 工具覆盖 | 无（grading）；MCP 仅在跑生成流水线时需要 | **✅ 完全可跑** |
| **ResearchClawBench** | **57/57**（代理打分所需） | ✅ checklist 文本加权代理 | 官方需要多模态 LLM judge | **⚠️ 仅代理可跑** |
| **SciPredict** | **405/405 GTA**；131/405 rubric（仅 FF） | ✅ MCQ/Numerical 确定性；FF 代理 | 官方 FF 需要 LLM judge | **⚠️ MCQ+Num 可跑；FF 仅代理** |
| **SMDDBench** | **502/502 task.yaml**；`eval_actives/inactives` 仅 25/502 | ❌ 永远返回 0.0（除非有 `result.json`） | Boltz-1 模型 + Docker + 上游 repo | **❌ 没有上游 repo、没有权重** |

### 1. BiomniEval1

- **GT**：parquet 中 `answer` 字段 433/433 行全部非空，10 个 task 家族都覆盖到。
- **验证代码**：`grade_biomni_eval1_answer`（`grader.py:795-873`），纯 stdlib —— 正则抽取 MCQ 字母、基因 token 包含、JSON 解析 OMIM/causal_gene。无需 MCP、无需代码执行、无需网络。
- **外部依赖**：无。哪怕上游 Biomni 的 `biomni_eval1.py` 也只是 `hf://datasets/biomni/Eval1/...` 形式，本地完全离线。
- **结论**：六个里**最坚实的一个**。

### 2. DrugDiscoveryBench

- **GT**：82 个任务目录里**全部 `rubrics.json` 都是占位** —— `ground_truth=""`、`outcome_rubrics=[]`、`process_rubrics=[]`，只有 `prompt` 有内容。
- **验证代码**：本地 `grade_drug_discovery_bench_answer`（`grader.py:456`）在 `outcome_rubrics` 为空时返回 `(False, 0.0, [...populate_rubrics.py...])`。每个任务目录下都有一份 `tests/judge.py`（476 行，LiteLLM）+ `tests/test.sh`，**官方路径不调用这些**（仅 Harbor 容器内的 LLM judge 调用）。
- **外部依赖**（很重）：
  - Docker 镜像 `ghcr.io/scaleapi/drugdiscoverybench:1.0.0-lightweight`（~6 GB 压缩，~23 GB 解压）
  - Conda env `biomni_e1`、Node 22、agent CLIs（`claude-code`/`codex`/`gemini-cli`/`grok`）
  - Gated HF 数据集 `ScaleAI/DrugDiscoveryBench`（HF_TOKEN；自动审批）
  - Agent LLM keys + Judge LLM key + `BRAVE_SEARCH_API_KEY`
  - Harbor 框架 (`harbor==0.13.1`)
- **修复路径**：
  1. 把 `/Users/a123/Documents/games/DrugDiscoveryBench-main/scripts/populate_rubrics.py` 拷贝到 `/Users/a123/Documents/games/sciMAS/scripts/`（当前只有 `biomni_codegen.py` 和 `sync_dataset.sh`）
  2. `populate_rubrics.py --token $HF_TOKEN` 把 82 份 rubrics 就地填好
  3. （可选）接入 82 个 `tests/judge.py` 做 process rubric 评分

### 3. MADD

- **GT**：100/100 行 `tools answers`（字符串化的 markdown SMILES 表）、`finall answer`、`is_correct`（87/100 为 1）全部填充。44 个不同的 `case` 组合。
- **验证代码**：`grade_madd_answer`（`grader.py:331-361`）正则抽取黄金 SMILES，`grade_madd_tool_coverage`（`grader.py:406-443`）对照 `gen_mols_<disease>` 工具名做 case-code 别名匹配。纯 Python，无需化学后端。
- **外部依赖**：grading 无依赖。`pharma-drug-discovery` MCP 仅在真正执行分子生成流水线时才用。MADD-main 没有独立 grader，上游 SSA 逻辑内联在 `multi_agent_system/MADD_main/MADD_run_on_benchmark.py`。
- **结论**：六个里**仅次于 BiomniEval1 的坚实度**。

### 4. ResearchClawBench

- **GT**：`hf_dataset/train/data-00000-of-00001.arrow` 57 行；57 个 `target_paper_path` 文件全部存在；57 份 `checklist_json` 都填充；8259/8259 data files + 228/228 related-work PDFs + 159/159 target images，总载荷 ~2.47 GB。
- **验证代码**：`grade_checklist_answer`（`grader.py:204-257`）做 checklist 文本/关键词加权（≥0.5 通过），确定性、自包含。
- **外部依赖**：**仅官方打分需要** —— vendored `repo/evaluation/` 里**只有 17 个 vendor logo**（`static/logos/`），**没有 judge 脚本**；`eval.yaml` 写明 `evaluation_framework: none`。官方 judge 在外部 HF Space (`ResearchHarness`) + `rcb-eval` CLI，**都没有 bundled**。
- **运行痕迹**：`tests/results/ResearchClawBench/20260913-125103-rcb-5-net/` 之前有过 5 题 80% 答案准确率 / 0.58 checklist 平均分 / ~32 min 的运行。
- **结论**：57 题代理打分完全可跑；官方 leaderboard 需要网络 + LLM judge。

### 5. SciPredict

- **GT**：`main_ds.csv` 405/405 任务都有 `GTA`、404/405 有 `CLEAN_GTA`、100% 有 `REQUIRED_BACKGROUND_KNOWLEDGE`、400/405 有 `RELATED_PAPERS_DATA`。rubrics.csv **131/405（32.3%）** —— 只覆盖 131 道 Free-Format 题；MCQ (162) + Numerical (112) 没有 rubric，但走 `CLEAN_GTA` 评分。
- **验证代码**：`grade_scipredict_answer`（`grader.py:584-648`）三条确定性路径：MCQ 字母集合相等（精确 1.0 或部分匹配比例）、Numerical 区间正则（落在任一区间即满分）、Free-Format 走 `grade_checklist_answer` 代理（注释明确"official scoring uses a judge"）。
- **外部依赖**：本地 grading 无。Scale leaderboard 官方分数需要 LLM judge。`/Users/a123/Documents/games/` 下没有 `scipredict/` 或 `ScaleAI/` clone，README 列的所有回退路径都是死链。
- **修复路径**：
  1. 那一行缺失的 `CLEAN_GTA` 应该 triage
  2. `test_scipredict_dataset.py` 用的是 3 行合成 fixture，**没有真测** 405 行 CSV —— 应加一个直接 load `dataset/SciPredict/data/main_ds.csv` 的 smoke test
  3. FF 代理分数与 Scale 公开数字不可直接比较，建议在报告里标注

### 6. SMDDBench

- **GT**：502/502 `task.yaml` `evaluation` 块全部填写；但 `solution.smi`（选手样本）0/502；`eval_actives.smi`/`eval_inactives.smi` 仅在 25/502 个 `smdd_001_*` 任务里存在；仓库内**没有任何 `result.json`**。任务分布：001=25，002=25，003=52，004=340，005=60。
- **验证代码**：**没有** —— `grade_smdd_bench_answer`（`grader.py:727-763`）只读取任务目录里的 `result.json`（或内嵌 `smdd_official_result`）；没有则直接返回 `(False, 0.0, [...Run the upstream SMDD-Bench Docker evaluator...])`。**不会**从文本重新计算指标（docstring 里明确写了）。
- **外部依赖**（**最重**）：
  - **452/502（~90%）任务需要 Boltz-1**（003/004/005 用 `metrics.interaction.assess_boltz_all`、`metrics.lead_optimization.assess_boltz_task4`、`metrics.fragment_assembly.assess_boltz_task5`）
  - `task.yaml` 引用的 `metrics.*` 模块**全磁盘上不存在**（全局 grep `metrics.pharmacophore` / `metrics.scaffold` / `metrics.interaction` / `metrics.lead_optimization` / `metrics.fragment_assembly` / `metrics.chemistry` 都是 0 命中）
  - **Boltz-1 权重没有缓存**（`~/.cache/`、`dataset/`、`/Users/a123/Documents/games/` 都没有 `*.ckpt` / `*.pt` / `*.safetensors`）；`python3 -c "import boltz"` → ModuleNotFoundError
  - **上游 repo 没有**（`/Users/a123/Documents/games/SMDD-Bench/`、`t7rs/`、`SMDDBench-main/` 全部缺失）
- **修复路径**（**工作量大**）：
  1. clone `https://github.com/t7rs/SMDD-Bench`
  2. `pip install boltz`
  3. 下载 Boltz-1 权重（多 GB）
  4. 把上游 `metrics/*.py` 接到 sciMAS 的 grader（或跑上游 Docker 评估器，再把 `result.json` 喂回来）

### 跨数据集发现

- **离线可跑性排序**：`BiomniEval1 (433) ≈ MADD (100)` > `ResearchClawBench (57, 代理) ≈ SciPredict MCQ+Num (274)` >> `DrugDiscoveryBench (需 populate_rubrics) ≈ SMDDBench (需 Boltz + Docker)`
- **测试偏弱**：仅 DrugDiscoveryBench 有单元测试覆盖"空 rubric → 0.0"；MADD 3 个用例；BiomniEval1 2 个；SciPredict 用 3 行合成 fixture；SMDDBench 用合成 fixture。**没有一套测试用真实 bundled 数据跑端到端**。
- **`/Users/a123/Documents/games/` 下没有上游 repo clone** for SMDD-Bench、SciPredict、ResearchClawBench；只有 `Biomni/`、`DrugDiscoveryBench-main/`、`MADD-main/`、`SciAgentGYM-main/`、`Skill-Pro/`、`sciMAS/`。
- **没有任何模型权重预缓存**：除了 `agent_harnesses.json` 之外没有 Boltz-1、HF model blobs 等。

### 关键引用文件（审计相关）

- `/Users/a123/Documents/games/sciMAS/dataset/{BiomniEval1,DrugDiscoveryBench,MADD,ResearchClawBench,SciPredict,SMDDBench}/`
- `/Users/a123/Documents/games/sciMAS/tests/grader.py`、`/Users/a123/Documents/games/sciMAS/tests/dataset.py`
- `/Users/a123/Documents/games/DrugDiscoveryBench-main/scripts/populate_rubrics.py`（需要 vendoring 到 `/Users/a123/Documents/games/sciMAS/scripts/`）
- `/Users/a123/Documents/games/Biomni/biomni/eval/biomni_eval1.py`（上游参考 grader）