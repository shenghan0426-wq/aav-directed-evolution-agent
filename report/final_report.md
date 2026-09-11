# 面向蛋白质定向进化的科学智能体：基于 AAV 突变数据的虚拟实验研究

## 摘要

蛋白质定向进化需要根据已有实验结果设计下一轮突变体，但蛋白质序列空间巨大，多位点突变之间存在上位性效应，导致随机筛选和人工设计效率有限。本文以 AAV 突变 fitness 数据为对象，构建一个轻量级科学智能体，用于模拟“历史实验分析、突变假设生成、候选序列设计、适应度模型打分、虚拟实验反馈”的定向进化流程。系统结合 ESM-2 蛋白质语言模型、MLP fitness head、大语言模型 Agent 和氨基酸性质知识库，在测试候选池上进行三轮虚拟定向进化实验。结果显示，ESM2-MLP 在测试集上取得 Pearson = 0.617、Spearman = 0.616，但最高分候选仍存在明显高估。原始 LLM Agent 倾向于组合过多历史高频突变，真实 fitness 表现较差；知识增强后，推荐候选平均突变数从 17.4 降至 4.0，平均真实 fitness 从 -4.743 提升至 3.282。结果表明，LLM Agent 可以模拟科研流程，但可靠推荐需要适应度模型、知识规则和迭代反馈共同约束。

## 1. 引言

蛋白质定向进化是酶工程、抗体优化和合成生物学中的重要方法。其核心思想是从已有变体的实验结果出发，设计下一轮突变库，筛选高 fitness 变体，并进行多轮迭代优化。然而，蛋白质序列空间随位点数呈指数增长，不同突变之间还可能发生正向或负向上位性，使得单点突变效果不能简单推广到多点组合。

近年来，蛋白质语言模型和大语言模型 Agent 为定向进化提供了新的计算框架。蛋白质语言模型可学习序列上下文表示，适应度预测模型可从历史突变数据中近似 fitness landscape，而 LLM Agent 可模拟科学家的研究流程：读取实验数据，归纳关键位点，提出突变假设，设计候选序列，并解释推荐理由。本文关注的问题是：轻量级 LLM Agent 是否能在 AAV 突变数据上形成有效的推荐流程，以及知识增强是否能改善其推荐质量。

## 2. 数据集与任务定义

本文使用 AAV two-vs-many 突变数据。训练集被视为已经完成的历史实验结果，测试集被视为尚未实验验证的候选突变池。每条样本包含全长蛋白质序列和连续 fitness 分数。代码从全长序列中提取 AAV 关键突变区域，并表示为相对于野生型的氨基酸替换列表。

野生型突变区域为：`DEEEIRTTNPVATEQYGSVSTNLQRGNR`。

历史高 fitness 样例包括：`V579Y/R585Q`，fitness = 9.536；`Q575A/T581E`，fitness = 8.645；`S578I/T581D`，fitness = 8.438；`Q575W/R588Q`，fitness = 8.390；`V579A/T581E`，fitness = 8.078。这些样例提示 575、578、579、581、585、588 等位点可能与高 fitness 相关。

实验设置为：Round 0 使用训练集作为已有实验结果；Round 1 至 Round 3 中，不同策略从测试候选池中推荐 Top-k 变体，并用测试集真实 fitness 作为虚拟实验测量值。每轮被评估的候选会加入对应策略的历史数据，用于下一轮推荐。

## 3. 方法

### 3.1 适应度预测模型

适应度预测模块采用 ESM-2 提取蛋白质序列 embedding，并使用 MLP fitness head 将 embedding 映射到连续 fitness。为了判断模型是否优于简单基线，本文比较训练集均值 baseline、one-hot Ridge regression 和 ESM2-MLP。

|model|MSE(test)|Pearson|Spearman|Top-10 hit|PredTop10 true mean|
|---|---|---|---|---|---|
|train_mean_baseline|12.794|-0.000||0.000|-0.806|
|one_hot_ridge|59.570|-0.032|0.006|0.000|-3.552|
|esm2_mlp_fitness_head|5.565|0.617|0.616|0.000|-3.220|

ESM2-MLP 在测试集上达到 Pearson = 0.617、Spearman = 0.616，说明蛋白质语言模型表征能够捕捉部分 fitness landscape 信号。然而，Top-10 hit rate 为 0，预测 Top-10 的真实均值为 -3.220，说明模型在最高 fitness 区域仍有高估风险。

### 3.2 LLM Agent 架构

LLM Agent 采用模块化 prompt + Python 函数调用方式实现，包括五个模块：Data Analyst 读取训练集历史实验结果和 top variants，总结高 fitness 样本中的富集突变；Hypothesis Generator 生成可检验突变假设；Mutation Designer 从候选池中选择下一轮候选；Fitness Evaluator 调用 ESM2-MLP 预测 fitness；Scientific Critic 综合历史证据、预测分数和不确定性，输出 Top-k 推荐及理由。

该设计中，LLM 不直接编造 fitness，而是负责科学分析、假设生成、候选选择和解释。数值打分来自适应度预测模型；虚拟实验评价来自测试集真实 fitness。

### 3.3 知识增强

知识增强模块包含氨基酸理化性质、保守替换规则和突变数量约束。理化性质包括电荷、极性、大小和疏水性；替换类型被划分为 conservative substitution 和 radical substitution；规则库还对过多突变的候选施加惩罚，并检查异常氨基酸或非法突变格式。

系统还构建了简单知识图谱三元组，例如 `Amino Acid -- has_property -- Hydrophobic`、`Mutation -- occurs_at -- Position`、`Mutation -- observed_with_fitness -- Fitness`、`Variant -- contains -- Mutation`。知识增强得分由历史突变先验和规则分数归一化加权得到。

|strategy|mean mutations|mean predicted|mean true|best true|mean pred error|
|---|---|---|---|---|---|
|after_knowledge|4.000|2.308|3.282|5.999|-0.974|
|before_knowledge|17.400|-2.521|-4.743|-4.136|2.222|

知识增强后，候选平均突变数从 17.4 降至 4.0，平均真实 fitness 从 -4.743 提升至 3.282，说明在本任务中控制突变复杂度比盲目组合历史高频突变更重要。

## 4. 虚拟定向进化实验结果

本文比较四种推荐策略：随机候选选择、适应度模型直接推荐、原始 LLM Agent 推荐和知识增强 LLM Agent 推荐。每轮每种策略推荐 Top-k = 10 个候选，并用测试集真实 fitness 作为虚拟实验结果。

|strategy|round|mean fitness|best fitness|mean improved|best improved|
|---|---|---|---|---|---|
|fitness_model_direct|1|0.176|4.088|False|False|
|fitness_model_direct|2|3.228|6.867|True|True|
|fitness_model_direct|3|0.504|5.952|False|False|
|knowledge_enhanced_llm_agent|1|3.282|5.999|False|False|
|knowledge_enhanced_llm_agent|2|0.925|4.269|False|False|
|knowledge_enhanced_llm_agent|3|2.011|6.262|True|True|
|llm_agent|1|-4.235|-2.631|False|False|
|llm_agent|2|-4.666|-4.008|False|False|
|llm_agent|3|-4.506|-3.648|True|True|
|random_mutation|1|-1.873|3.419|False|False|
|random_mutation|2|-1.968|3.691|False|True|
|random_mutation|3|0.597|4.368|True|True|

适应度模型直接推荐在 Round 2 达到最高平均 fitness 3.228 和最高 best fitness 6.867，但 Round 3 回落，说明模型排序有用但不稳定。原始 LLM Agent 三轮平均 fitness 均为负，低于随机策略，主要原因是其倾向于组合大量历史上看似有益的突变。知识增强 LLM Agent 在 Round 1 平均 fitness 达到 3.282，Round 3 best fitness 达到 6.262，显著优于原始 LLM Agent。

## 5. 每轮 Agent 推荐结果

完整 Top-k 推荐结果保存在 `artifacts/virtual_evolution_topk_by_round.csv`。下表展示原始 LLM Agent 与知识增强 LLM Agent 每轮推荐中真实 fitness 最高的候选。

|strategy|round|Top-k candidate IDs|best candidate|best mutations|best true fitness|mean true fitness|
|---|---|---|---|---|---|---|
|llm_agent|1|C02684, C44468, C42789, C42787, C42790, C02685, C49503, C44605, C44604, C44469|C44605|D561M,E563N,Q575V,Y576F,S578I,T581E,N582T,L583P,Q584D,R585F,G586F,N587M,R588S|-2.631|-4.235|
|llm_agent|2|C42276, C42274, C42269, C42273, C42265, C42266, C42264, C42267, C42272, C42270|C42270|D561G,E563L,I565L,R566V,T567C,T568V,V571T,A572C,Q575L,Y576W,S578D,V579A,S580A,T581E,N582E,Q584M,R585F,G586Y,N587L|-4.008|-4.666|
|llm_agent|3|C42271, C46538, C42275, C00984, C00987, C00988, C30224, C00985, C37776, C00982|C30224|I565L,R566A,T567G,Y576F,S578D,V579A,S580T,T581E,N582I,R585A,G586P,N587G|-3.648|-4.506|
|knowledge_enhanced_llm_agent|1|C11943, C16318, C11940, C15184, C16328, C17800, C15198, C15827, C15832, C11936|C16318|V579A,T581E,L583I,R588A|5.999|3.282|
|knowledge_enhanced_llm_agent|2|C05535, C05536, C05513, C05514, C05545, C05529, C05530, C30356, C30333, C03345|C05545|R566C,V579A,T581D,N587G|4.269|0.925|
|knowledge_enhanced_llm_agent|3|C42630, C15475, C15477, C16099, C15888, C16100, C16123, C50117, C15837, C16292|C15837|S578E,V579A,N587T|6.262|2.011|

## 6. 关键位点集中度分析

推荐突变并非均匀分布，而是集中在少数位点。下表列出每种策略每轮 Top-k 中突变频率最高的位点，括号内为该位点在当轮 10 条推荐中的出现频率。

|strategy|round|top recurrent positions(freq)|
|---|---|---|
|fitness_model_direct|1|578(0.9), 587(0.8), 561(0.7)|
|fitness_model_direct|2|561(0.9), 581(0.9), 587(0.8)|
|fitness_model_direct|3|587(1.0), 578(0.8), 579(0.8)|
|knowledge_enhanced_llm_agent|1|587(0.8), 579(0.6), 581(0.6)|
|knowledge_enhanced_llm_agent|2|566(0.8), 579(0.7), 581(0.7)|
|knowledge_enhanced_llm_agent|3|578(0.8), 581(0.8), 579(0.5)|
|llm_agent|1|563(1.0), 575(1.0), 578(1.0)|
|llm_agent|2|561(1.0), 563(1.0), 565(1.0)|
|llm_agent|3|565(1.0), 566(1.0), 567(1.0)|
|random_mutation|1|581(0.8), 588(0.8), 566(0.6)|
|random_mutation|2|566(0.8), 578(0.8), 585(0.8)|
|random_mutation|3|578(0.7), 581(0.7), 588(0.6)|

结果表明，578、579、581、587、588 等位点在多个策略中反复出现，与历史高 fitness 样本中的热点位点基本一致。原始 LLM Agent 对热点位点的集中度很高，但没有充分控制组合复杂度；知识增强 LLM Agent 则在保留热点位点的同时限制突变数量，推荐质量更好。

## 7. 成功案例与失败案例

知识增强 LLM Agent 的成功案例包括：

|candidate_id|mutations|predicted_fitness|true_fitness|prediction_error|
|---|---|---|---|---|
|C16318|V579A,T581E,L583I,R588A|3.387|5.999|-2.612|
|C15827|S578E,V579A,S580A,N587G|4.403|5.952|-1.549|
|C15832|S578E,V579A,T581E,N587M|4.416|5.630|-1.214|

这些候选通常只含 3-4 个突变，围绕 S578、V579、T581、N587/R588 等热点位点组合，说明局部、低复杂度组合更可能保留有益效应。

原始 LLM Agent 的失败案例包括：

|candidate_id|mutations|num_mutations|predicted_fitness|true_fitness|prediction_error|
|---|---|---|---|---|---|
|C02684|E563A,R566A,T568V,A572C,Q575V,S578D,V579A,S580D,T581D,N582A,L583I,Q584C,R585F,R588F|14|0.108|-5.739|5.847|
|C02685|E563A,R566A,T568V,A572C,Q575V,S578D,S580D,T581D,N582A,L583I,Q584C,R585F,R588F|13|0.165|-5.731|5.897|
|C42787|D561H,E563L,I565L,R566A,T567C,E574S,Q575C,S578D,V579A,S580A,T581E,N582E,L583I,R585T,N587M,R588M|16|1.320|-5.059|6.379|

失败候选通常包含 13 个以上突变。尽管部分突变来自历史高 fitness 样本中的热点位点，但复杂背景可能引入强烈负向上位性，导致真实 fitness 显著低于预测值。

## 8. 讨论：LLM Agent 是否学到了科学家思维

从流程角度看，LLM Agent 确实模拟了科学家的部分研究步骤：它读取历史实验结果，发现可能重要的位点，提出突变假设，选择下一轮候选，并给出可读推荐理由。然而，从实验结果看，原始 LLM Agent 更多是在组织已有统计信号，并依赖预测模型排序，而不是真正理解突变组合的因果机制。它能够发现热点位点，却不能自动判断哪些组合会因背景突变过多而失败。

知识增强版本更接近科学家的保守实验设计方式：优先选择低突变数、机制上可解释、围绕热点位点的候选，并将失败解释为背景依赖或上位性风险。因此，LLM Agent 的价值主要在于科研流程组织和可解释假设生成；要获得可靠推荐，仍需适应度模型、知识规则、知识图谱和迭代反馈共同约束。

## 9. 交互式 Demo

本文实现了一个简单交互式 demo。用户可以输入 AAV 野生型突变区域，设置 Top-k 推荐数量和最大突变数，系统会读取训练集历史实验结果、测试候选池和已有模型预测文件，输出候选突变、预测 fitness、知识增强得分和推荐理由。该 demo 主要用于展示智能体如何将历史数据、适应度模型和知识规则整合为可交互的突变推荐流程。

运行命令为：

```bash
cd aav_baseline
python -m streamlit run app_demo.py
```

## 10. 结论与未来工作

本文构建并评估了一个面向 AAV 蛋白定向进化的轻量级科学智能体。实验表明，ESM2-MLP 能提供有用但不充分的 fitness 预测信号；原始 LLM Agent 可以生成科学解释，但容易过度组合历史高频突变；知识增强显著降低突变复杂度，并提高推荐候选的真实 fitness。未来工作可以引入主动学习策略、Gaussian Process 或模型集成不确定性估计、结构信息、保守位点分析，以及与自动化实验平台连接的闭环优化系统。

## 11. 使用资源说明

本项目使用公开 AAV 突变 fitness 数据、ESM-2 蛋白质语言模型、OpenAI API，以及 pandas、scikit-learn、PyTorch、matplotlib、Streamlit 等 Python 工具。项目代码整理、notebook 流程组织和报告草稿撰写使用了 ChatGPT/Codex 辅助。
