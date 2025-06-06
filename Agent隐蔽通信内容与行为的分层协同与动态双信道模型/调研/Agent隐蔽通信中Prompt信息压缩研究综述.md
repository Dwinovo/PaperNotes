## 一、 引言：Agent隐蔽通信的挑战与Prompt的角色

"Agent隐蔽通信内容与⾏为的分层协同与动态双信道模型"。其核心思想是构建两个并行的通信信道：
- 一个高带宽的"内容信道"，负责传输主要的、大容量的秘密信息；
- 一个低带宽但关键的"行为信道"，利用Agent在环境中的交互行为模式（如消息发布频率、时间间隔、对特定内容的互动等）来传递控制和同步信号。
这种分层协同机制，特别是通过行为信道传递Prompt变更等元信息，旨在解决内容信道的静态Prompt同步瓶颈。

在此双信道模型中，行为信道的带宽通常极为有限。因此，若要通过行为信道隐蔽、可靠地传输Prompt本身或其变更指令（例如，切换到新的Prompt模板，或对现有Prompt进行微调的参数），对Prompt信息进行高效的压缩与精确的解压缩便成为至关重要的核心技术前提。未经压缩的Prompt信息量巨大，远非低带宽的行为信道所能承载。因此，研究适用于此类场景的Prompt压缩与解压缩方法，对于实现灵活、鲁棒且自主的Agent隐蔽通信具有决定性意义。

Prompt 压缩需要关注**三大维度**：
1. **尺寸与重量**：适应行为信道极低带宽的轻量化设计，将复杂 Prompt 转化为少量比特流 。
2. **装药与引信**：精确编码与可靠解码，确保低带宽信号能准确承载 Prompt 变更指令，并被接收方Agent无误解码 。
3. **威力**：恢复后的 Prompt 能准确指导 Agent 执行任务，关注保真度和行为一致性 。
## 二、 主流Prompt压缩技术分类、原理与评估

### 1. 索引化与查表方法 (Indexing & Look-up Table)

**基本原理**：预先在通信双方共享一个静态或可扩展的Prompt模板库，每个模板赋予唯一标识符（如数字ID或哈希值）。通信时，仅传递所需Prompt的标识符，接收方查表获取完整内容。

**关键技术细节**：
- **静态查表法**：最简单的形式，预设固定Prompt集合，双方在通信前就达成共识。例如，《行为模式-弹道选择》中提到的"基于共享内存的交互模式选择"，可预先分配Prompt表，通过特定行为模式（如，选择不同交互对象的顺序）来实际传递Prompt ID。
- **动态字典构建**：借鉴LZ系列压缩算法思想，随着交互进行逐步构建共享字典。通过在内容信道传输的文本内容中隐含地增加新Prompt模板，然后通过行为信道传递使用指示。
- **层级索引系统**：对Prompt按功能类别、应用场景等进行层级分类，形成树状索引结构，减少传输所需比特数，且便于系统化管理和扩展。
- **哈希映射优化**：利用哈希函数将Prompt映射至紧凑标识符，配合避免碰撞的设计，可支持极大的可寻址空间。

**优势**：
- 极高的压缩率，仅需传递极少比特即可表示复杂Prompt
- 零解压缩误差，完美还原预设Prompt
- 解码速度极快，几乎无计算延迟

**局限性**：
- 缺乏灵活性，难以传递全新或微调的Prompt
- 预设库规模与更新是瓶颈，大库难以隐蔽同步
- 可能通过统计分析发现异常选择模式

**适用场景**：适合预定义任务集有限、变化性低的隐蔽通信，特别是对行为信道带宽极为受限的情况。若配合参数化模板使用（见下节），可大幅提升灵活性。

### 2. 参数化模板与差分编码 (Parameterized Templates & Differential Encoding)

**基本原理**：参数化模板将Prompt结构化为基础模板+可变参数，仅传递变化的参数值大幅减少信息量；差分编码则只传递相对上一版本Prompt的变化部分。

**关键技术细节**：
- **参数化Prompt模板**：设计包含变量槽位的模板（如`{role}`执行`{action}`的任务，其中`{constraint}`），仅传递变量值（role="研究者", action="分析数据", constraint="24小时内"）。变量槽可预定编号，极大压缩信息。
- **插槽设计优化**：针对特定领域，研究最优的变量槽设计和粒度，平衡表达能力与压缩率。例如，《双信道模型》中提到的"精确语法"与"平行语义"变量粒度选择策略。
- **差分编码技术**：类似Git版本控制，记录基准Prompt与目标Prompt的差异。支持多种差分计算方法：
  - 文本级差分：记录添加/删除/修改的文本段
  - 标记级差分：更细粒度地跟踪Token变化
  - 语义差分：基于嵌入空间的语义变化描述
- **变更指令压缩**：将差分操作本身编码为极简指令（如"在第3行'执行'后插入'迅速'一词"）

**优势**：
- 显著的压缩率提升，尤其对结构化场景
- 保持高度灵活性，支持新Prompt构建
- 易于适配变化频繁的动态环境

**局限性**：
- 需预共享模板库或基准版本
- 参数化设计对语言和表达的限制
- 差分编码对基准版本的依赖性高

**适用场景**：适合结构化程度高、变化相对有限的Prompt传递。在基于"模式-弹道选择"的隐蔽通信中，可优化配合时间型或交互型行为信道，使用有限比特传输关键参数变更。

### 3. 基于模型的Prompt信息压缩方法

**基本原理**：将Prompt转换为语义向量空间中的表示，利用降维技术压缩后传输，接收方重构向量并生成等效Prompt。这种方法依托于大型语言模型内在的语义理解能力。
#### 1. 500xCompressor: Generalized Prompt Compression for Large Language Models (论文1)
- **实现了前所未有的高压缩率**：
    - **创新点**：该研究提出了500xCompressor，能够将自然语言上下文压缩到最少一个特殊token，实现了6x到480x的压缩率 。这是目前文献中报告的最高压缩率之一，远超先前方法（如ICAE的压缩率不超过15x ）。
    - **为什么提出/贡献**：长提示（Long prompts）会显著降低推理速度、增加计算成本并影响用户体验，同时上下文长度限制也制约了模型的应用场景 。已有的压缩方法在压缩率上存在瓶颈 。此项创新通过大幅提升压缩率，极大地缓解了这些问题，为更高效、经济地使用大语言模型铺平了道路。
- **关键技术改进：使用KV值而非嵌入（Embeddings）进行信息编码**：
    - **创新点**：与ICAE等先前主要依赖压缩token嵌入（embeddings）的方法不同，500xCompressor将文本信息编码到压缩token在LLM每一层中的键值（KV）对中，并将这些KV值传递给解码器 。
    - **为什么提出/贡献**：论文指出，KV值相比嵌入能够封装更多信息，且对推理时间和GPU内存影响较小 。这一技术选择被认为是其在高压缩率下仍能较好保留信息的关键因素之一 ，从而在文本再生和问答任务上超越了基线方法ICAE 。这为软提示压缩提供了一个更有效的信息载体。
- **保持原始LLM能力且无需微调解码器**：
    - **创新点**：压缩后的token可以直接被原始的、未经微调的LLM用于下游任务（如问答或文本再生） 。训练过程中，仅编码器中的LoRA参数是可训练的，原始LLM参数在编码器和解码器中均保持冻结 。
    - **为什么提出/贡献**：许多早期的软提示方法（如GIST）需要微调原始LLM才能使用压缩提示 。这不仅增加了部署的复杂性和成本，还可能损害LLM原有的通用能力。500xCompressor通过避免微调解码器，保留了LLM的原始能力，并极大地方便了压缩token的使用和集成 。
- **严格的评估体系与对数据泄漏的关注**：
    - **创新点**：研究者特别关注了评估过程中的潜在数据泄漏问题，采用了严格未见（strictly unseen）的评估集（Arxiv语料和ArxivQA数据集中的文本均在LLaMa-3系列模型知识截止日期之后发布） 。同时，采用抽取式问答（extractive QA）进行定量信息损失分析，答案直接来源于上下文，有明确的目标答案 。
    - **为什么提出/贡献**：先前一些研究的评估文本可能与LLM的训练数据重叠，导致难以判断模型表现是源于压缩信息的理解还是LLM的记忆 。通过使用严格未见数据和可定量的评估指标，500xCompressor的评估结果更具说服力，能更真实地反映压缩方法本身的性能和信息保持能力。
- **通用性与对复杂信息的处理能力**：
    - **创新点**：500xCompressor被设计为可以压缩任何文本，并用于回答各种类型的问题，展示了其泛化能力 。分析表明，即使是专有名词、特殊名称和数字等复杂信息也能被准确压缩和检索 。
    - **为什么提出/贡献**：一些硬提示方法是有选择性的，可能会丢失部分信息 。而500xCompressor旨在再生整个原文，确保所有原始文本token都对压缩token有贡献 。这种通用性和对细粒度信息的处理能力，拓展了提示压缩技术的适用范围。
* **核心特点**：利用LLM自身进行端到端压缩，将自然语言上下文压缩至极少数（最少一个）特殊token [cite: 3]。
* **信息表示**：关键信息主要存储在这些特殊token的**KV值 (Key-Value values from attention layers)**中，而非仅仅是它们的嵌入 [cite: 8]。
* **模型修改**：编码器LLM使用冻结参数并配备可训练的LoRA参数进行训练 [cite: 12, 44]。解码器则使用原始冻结的LLM，无需额外微调 [cite: 12, 44]。
* **训练方式**：包含预训练（文本再生任务）和指令微调（问答任务）两个阶段 [cite: 13, 15]。
* **优势**：实现了极高的压缩率（6x-480x） [cite: 4]，压缩后的提示可直接被原始LLM使用 [cite: 5]，对复杂信息（如专有名词、数字）有较好的压缩和检索能力 [cite: 27]。
[[500xCompressor Generalized Prompt Compression for Large Language Models]]
#### 2. AN EMPIRICAL STUDY ON PROMPT COMPRESSION FOR LARGE LANGUAGE MODELS (论文2)
* **核心特点**：这是一篇**实证研究**，本身不提出新的压缩方法，而是对现有的六种提示压缩方法（KiS, SCRL, Selective Context, LLMLingua, LongLLMLingua, LLMLingua-2）进行多维度综合评估 [cite: 3, 43, 55, 88]。
* **评估维度**：涵盖生成性能、模型幻觉、多模态任务中的效用、词语省略分析、响应长度等 [cite: 4, 30, 31]。
* **关键发现**：(Long)LLMLingua 和 LLMLingua-2总体表现较好，尤其在高压缩比下 [cite: 26, 54]。长上下文中适度压缩可能提升性能 [cite: 6, 7, 27]。所有方法都可能增加幻觉，信息丢失是主因 [cite: 29, 156]。
* **贡献**：提供了全面的比较基准和统一的开源工具包(PCToolkit) [cite: 7, 31]。
[[AN EMPIRICAL STUDY ON PROMPT COMPRESSION  FOR LARGE LANGUAGE MODELS]]
#### 3. Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt (论文3)
- **全新的视角：压缩后提示（Compress, Then Prompt）**
    - **创新点**：论文提出了一种新的范式，即在模型被压缩（如剪枝或量化）之后，再通过引入提示（prompt）来提升其性能，从而改善精度与效率之间的权衡 。这与传统思路——要么致力于改进压缩算法本身，要么在未压缩模型上进行提示调优以适应下游任务——有所不同。
    - **为什么提出/贡献**：模型压缩虽然能降低LLM的部署成本（如减少模型大小和推理延迟），但往往会牺牲模型质量 。研究者观察到，对于某些问题，精心设计的硬提示（hard prompts）能够显著改善压缩LLM的生成质量 。然而，手动设计普适有效的硬提示非常困难。此工作的贡献在于，它开辟了一条新的路径，即不满足于压缩后性能的必然下降，而是主动通过后续的提示学习来“修复”或“补偿”压缩带来的性能损失。
- **面向压缩模型的软提示学习方法**
    - **创新点**：基于硬提示的初步观察，论文提出了一种软提示学习（soft prompt learning）方法 。该方法的核心在于，在提示学习过程中，模型（的压缩权重）是可见的（exposed），目标是学习到能够增强压缩模型性能的提示 。这意味着学习到的提示能够感知到模型的压缩状态，并据此进行优化。
    - **为什么提出/贡献**：与之前主要利用提示使模型适应特定下游任务的提示调优框架不同 ，本文提出的可学习提示旨在普适性地提升压缩模型的整体性能，使其表现更接近未压缩的原始模型 。这为提升压缩LLM的实用性提供了一种数据驱动的有效手段。
- **可迁移的提示（Transferable Prompts）**
    - **创新点**：一个非常关键且新颖的发现是，这些为压缩模型学习到的软提示具有显著的可迁移性。它们不仅在单个模型和任务上有效，还可以迁移到：
        - 不同的数据集（Cross-Dataset Transferability）
        - 不同的压缩级别和压缩方法（Cross-Compression Transferability）
        - 不同的任务（Cross-Task Transferability）
    - **为什么提出/贡献**：这种可迁移性极大地增强了该方法的实用价值和效率。如果一个学习到的提示可以在多个场景下复用，就能显著减少为每个压缩模型或每个任务重新学习提示的开销 。论文甚至提出，这种迁移性使得可以将学习好的软提示“原位（in-situ）”地附加到新压缩的模型上，以在测试时提升其准确性 。这与大多数任务特定的提示调优形成了鲜明对比。
- **显著的性能提升效果**
    - **创新点**：实验结果表明，该方法能够极大地提升经过压缩的LLM（例如，8倍压缩的LLAMA-7B模型，结合了4位量化和50%权重剪枝）的性能，使其在流行的基准测试中能够匹敌未压缩的对应模型 。在某些情况下，例如使用SparseGPT进行50%稀疏度压缩或使用GPTQ进行4位量化的LLM，在加入软提示后甚至能超越完整模型的表现 。
    - **为什么提出/贡献**：这为在资源受限的硬件（如单个GPU）上部署高性能LLM提供了切实可行的方案 。它证明了“先压缩，再提示”的策略是有效改善精度-效率权衡的。
* **核心特点**：该论文的重点**并非压缩输入给LLM的Prompt**，而是通过学习一种**可迁移的软提示 (soft prompt)** 来提升**已经被模型压缩技术（如量化、剪枝）处理过的LLM**的性能 [cite: 111, 112]。
* **"Prompt"的角色**：这里的“提示”是一系列可学习的嵌入向量，附加在输入序列前，用于“纠正”或“补偿”因模型压缩带来的性能损失 [cite: 112, 116]。
* **压缩感知**：软提示的学习过程考虑了LLM的压缩状态，使其能适应并纠正压缩偏差 [cite: 114, 116]。
* **可迁移性**：学习到的软提示可以在不同数据集、任务和压缩级别/类型之间迁移 [cite: 112, 114]。
* **优势**：显著提升极端压缩LLM的性能 [cite: 112, 119]，且对推理延迟影响小 [cite: 115, 124]，无需微调LLM主体参数 [cite: 115]。
[[CompressThen PromptImproving AccuracyEfficiency  Tradeoff of LLM Inference with Transferable Prompt]]
#### 4. Covert Prompt Transmission for Secure Large Language Model Services (论文4)
- **开创性的问题定义与系统建模**：
    - **创新点**：论文首次系统地研究了LLM服务中的“隐蔽提示传输”问题 。作者构建了一个延迟最小化问题，该问题同时受到LLM响应的保真度（fidelity）和传输行为的不可检测性（detectability）的双重约束 。该模型还推导了在无线信道不确定性下隐蔽性要求的可处理表达式 。
    - **为什么提出/贡献**：随着LLM在云边架构中的广泛应用，通过无线方式传输的用户提示面临内容泄露和传输行为被侦测的风险 。尤其对于包含敏感信息的长提示，这些风险更为突出 。以往研究主要集中在LLM推理优化，而提示传输安全，特别是其隐蔽性，未得到充分探讨 。此工作填补了这一空白，为设计安全的LLM无线通信提供了理论基础和优化目标。
- **新颖的提示压缩与加密（PCAE）框架**：
    - **创新点**：提出了一个名为PCAE（Prompt Compression and Encryption）的轻量级框架，用于在传输前处理提示，以减少传输开销和保护查询机密性 。该框架包含两个主要部分：
        - **基于“惊异度”（Surprisal-guided）的提示压缩**：利用本地部署的小型语言模型（SLM）估计token级别的惊异度得分，选择性保留语义关键的token，丢弃冗余token 。
        - **轻量级基于置换的加密（Permutation-based Encryption）**：在压缩后对token序列进行加密，以混淆语义内容 。
    - **为什么提出/贡献**：长提示增加了传输时间和被侦测的风险 。PCAE通过SLM进行惊异度压缩，显著减少了计算开销和传输时长，同时保留了核心语义信息，使得LLM响应保真度与基线方法相当，但预处理延迟降低了五个数量级以上，从而支持实时边缘部署 。轻量级加密则在不引入过多计算负担的情况下增强了数据隐私 。
- **高效的隐蔽策略优化方法（GPPO）**：
    - **创新点**：为实现自适应的隐蔽传输，论文设计了一种基于组的近端策略优化（Group-based Proximal Policy Optimization, GPPO）的深度强化学习方法 。该方法通过以下机制联合优化提示压缩率和发射功率：
        - **多候选动作采样**：在每个状态下采样多个候选动作。
        - **组内最优选择**：在每个组内选择最优动作。
        - **KL散度惩罚**：引入KL散度惩罚项以提高策略的稳定性和探索能力 。
    - **为什么提出/贡献**：传统的优化方法在动态无线环境中可能表现不佳 。虽然已有DRL方法用于隐蔽通信，但它们通常忽略LLM驱动的服务和内容级安全（如提示加密） 。GPPO通过其独特的动作采样和更新机制，能够更有效地在多重约束（保真度、隐蔽性、延迟）下学习传输策略，相比现有强化学习策略，其隐蔽传输延迟降低高达38.6% 。
- **全面的性能验证**：
    - **创新点/贡献**：论文通过仿真实验验证了所提方法的有效性。PCAE框架在多种LLM主干（如DeepSeek-32B, Qwen-32B及其小版本）上均表现出良好的保真度和极低的预处理延迟 。GPPO方法在隐蔽传输效率上显著优于基线方法，能够在严格的保真度约束下实现更低的延迟 。
* **核心特点**：提出一个两阶段安全LLM服务方案，重点是隐蔽的提示传输，包含提示压缩与加密 (PCAE) 框架和基于深度强化学习 (GPPO) 的隐蔽无线传输优化 [cite: 126, 128, 129]。
* **PCAE压缩方法**：
    * 利用**本地部署的小型语言模型 (SLM)** 估计词元级别的**惊奇度 (surprisal) 分数** [cite: 3, 4, 129, 132]。
    * 选择性保留语义关键（高惊奇度）的词元，丢弃冗余词元 [cite: 4, 129, 133]。
    * 结合轻量级的基于置换的加密 [cite: 3, 129, 145]。
* **GPPO优化**：基于分组的近端策略优化 (GPPO)，用于联合优化提示压缩率和发射功率，以最小化传输延迟，同时满足保真度和隐蔽性约束 [cite: 5, 45, 129, 133]。
* **优势**：PCAE预处理延迟极低，适合边缘部署 [cite: 6, 49, 129]。GPPO能有效降低隐蔽传输延迟 [cite: 8, 50, 129]。
[[Covert Prompt Transmission for Secure Large  Language Model Services]]
#### 5. Dynamic Compressing Prompts for Efficient Inference of Large Language Models (LLM-DCP) (论文5)
- **基于马尔可夫决策过程（MDP）的动态压缩框架**：
    - **创新点**：论文首次将提示压缩明确地建模为一个马尔可夫决策过程（MDP） 。在这个框架下，一个名为DCP-Agent的智能体通过一系列序贯决策，逐步移除提示中的冗余token 。这种方法的核心在于它能够适应动态变化的上下文，并在压缩过程中保留关键内容 。
    - **为什么提出/贡献**：现有的一些任务无关的压缩方法主要基于信息熵等指标估计token重要性，这往往忽略了提示压缩的序贯特性，即每个token的重要性是依赖于不断变化的上下文的 。通过MDP建模，DCP-Agent能够在每一步决策时都考虑到当前压缩状态（即上下文），从而做出更优的token移除选择。这解决了传统方法在处理上下文依赖性方面的不足，旨在提升压缩的智能性和有效性 。
- **不依赖目标黑箱LLM的奖励函数设计**：
    - **创新点**：为了有效训练DCP-Agent，作者设计了一个精巧的奖励函数。该函数旨在平衡三个核心要素：压缩率、LLM输出的质量（通过与原始prompt输出的KL散度衡量）以及关键信息的保留度（通过BertScore衡量） 。一个关键的创新在于，计算LLM输出分布的KL散度时，并非直接使用目标黑箱LLM（这通常成本高昂），而是采用了一个经过指令微调的、与目标LLM输出分布对齐的小型模型（Ms​） 。
    - **为什么提出/贡献**：许多现有的基于强化学习的压缩方法或数据生成方法在训练过程中严重依赖目标黑箱LLM来提供奖励信号或生成大规模标注数据 ，这导致训练成本极高且不切实际 。LLM-DCP通过使用代理小模型来评估输出质量，显著降低了训练成本，增强了方法的实用性 。
- **分层提示压缩（HPC）训练策略**：
    - **创新点**：受到课程学习（curriculum learning）中逐步增加难度思想的启发 ，论文提出了一种分层提示压缩（Hierarchical Prompt Compression, HPC）的训练策略 。该策略通过逐步提升压缩任务的难度（例如，逐渐减小期望压缩率的范围 cs​,cl​ 和最大轨迹长度 Tmax​ ），使得DCP-Agent能够渐进式地学习如何在高效压缩和保护关键信息之间取得平衡 。
    - **为什么提出/贡献**：直接在高压缩率下训练智能体保留关键信息是一项挑战 。HPC策略通过由易到难的课程设置，使得智能体能够更有效地掌握压缩技巧，最终在保持信息完整性的同时实现有效的压缩 。实验表明，HPC策略的使用带来了压缩率和评估指标（如EM）的显著提升 。
- **任务无关性（Task-Agnostic）**：
    - **创新点**：LLM-DCP被设计为一种任务无关的提示压缩方法 。这意味着它旨在普适性地应用于各种下游任务，而无需为特定任务进行重新微调。
    - **为什么提出/贡献**：许多现有的压缩方法是任务相关的，通常需要针对特定任务进行微调，这限制了它们的通用性 。例如，LongLLMLingua需要根据问题动态调整压缩内容，可能不适用于摘要等任务 。任务无关性使得LLM-DCP更具灵活性和实用价值，尤其是在需要处理多种不同类型任务的场景中。
* **核心特点**：将提示压缩建模为**马尔可夫决策过程 (MDP)** [cite: 6, 148, 151]，由一个DCP-Agent（基于Transformer编码器）通过强化学习（PPO算法）顺序移除冗余词元 [cite: 6, 148, 151, 157]。
* **奖励函数设计**：平衡压缩率、输出质量（通过与一个经过“分布对齐”的小型语言模型比较KL散度）和关键信息保留（BERTScore） [cite: 7, 48, 153, 160]，**不依赖目标黑盒LLM进行训练监督** [cite: 8, 49, 153]。
* **训练策略**：引入**分层提示压缩 (HPC) 训练策略**，借鉴课程学习思想，逐步增加压缩难度 [cite: 9, 50, 153, 165]。
* **优势**：任务无关的动态压缩 [cite: 4, 151]，训练成本较低 [cite: 8, 49, 154]，对动态上下文适应性强 [cite: 6, 151, 250]，在高压缩率下性能优越 [cite: 10, 153, 167]。
[[Dynamic Compressing Prompts for Efficient  Inference of Large Language Models]]
#### 6. ICPC: In-context Prompt Compression with Faster Inference (论文6)
- **基于编码器的提示压缩机制**：
    - **创新点**：ICPC（In-context Prompt Compression）的核心思想是利用预训练的Transformer编码器（如BERT、RoBERTa等，参数量通常在百万级别）来计算提示中每个词的出现概率，并通过信息函数计算每个词所携带的信息量，从而指导压缩过程 。这与许多现有工作依赖LLM（参数量通常在十亿甚至百亿级别）进行压缩决策形成了鲜明对比 。
    - **为什么提出/贡献**：现有方法若使用LLM进行压缩，会带来显著的计算和内存开销 。Transformer编码器由于参数量远小于LLM，其推理速度可以快10到100倍 。同时，编码器的预训练使其能够有效捕捉和理解词语的上下文信息 。因此，ICPC的这一设计旨在大幅提升压缩过程的速度和效率，同时减少信息损失 。
- **特定于上下文的信息量计算与词语移除策略**：
    - **创新点**：ICPC通过一个定义的损失函数 L(xi​) 来量化移除某个词 xi​ 时的信息损失，该函数结合了词语的上下文概率 p(xi​∣xi,k​) 和与邻近词的某种相似度或关联性 sim(xi+n​,xi​)（如公式(1)所示）。基于计算出的损失值，ICPC采用一种自适应的过滤策略：首先对所有单元（词、短语或子句）的损失值进行排序，然后计算损失值的p-百分位数 Lp​，并移除所有损失值大于等于 Lp​ 的词汇单元 。
    - **为什么提出/贡献**：这种方法旨在通过量化信息损失来更精确地识别冗余内容，并通过动态调整阈值（p-百分位数）来灵活控制压缩程度，以在压缩和信息保留之间取得平衡 。这种基于编码器计算的上下文信息，而不是依赖LLM的判断，使得决策过程更高效。
- **分词单元的多粒度处理**：
    - **创新点**：ICPC在进行过滤时，不仅仅在词的层面操作，还在短语（phrase）和子句（clause）层面进行处理 。它将具有上下文嵌入的token分组为“分词单元”（participle units），这些单元可以是词、短语或子句，具体取决于所需的粒度 。
    - **为什么提出/贡献**：直接在词级别进行过滤可能无法捕捉到语言模式的细微结构 。通过在短语和子句等更大粒度上操作，模型可以在过滤过程中保留更丰富的语义和句法信息，从而更好地保持压缩后提示的含义完整性 。
- **实现了显著的压缩速度提升和良好的性能**：
    - **创新点/贡献**：实验结果表明，ICPC能够有效地压缩不同类别的长文本，并在多种自然语言处理任务上取得了更好的性能和压缩速度 。与基线方法（如Selective Context, LLMLingua）相比，ICPC在使用参数量小得多的编码器（如BERT）时，压缩时间显著减少（如表2所示，ICPC的训练/压缩时间远低于其他方法），同时在各项评估指标（如BLEU, ROUGE, BERTScore）上保持了有竞争力的性能，甚至在某些指标上有所提升 。
* **核心特点**：利用**预训练的Transformer编码器（非完整LLM）** 计算词元概率和携带的信息量（损失）来进行压缩 [cite: 14, 15, 180, 183]，旨在实现极快的压缩速度 [cite: 15, 180]。
* **信息量计算**：对每个“分词单元”（词、短语或子句） [cite: 20, 38, 184]，其移除损失综合考虑了对上下文连贯性的影响和该单元自身的条件概率 [cite: 41, 187, 191]。
* **过滤机制**：基于所有单元损失值的**p-th百分位数**设定动态阈值，移除损失值大于等于该阈值的单元 [cite: 43, 44, 184, 187]。
* **优势**：压缩过程本身速度极快（比依赖LLM的方法快10-100倍） [cite: 18, 183, 191]，计算和内存开销低 [cite: 3, 183, 192]，同时保持了较好的压缩性能和通用性（在多种编码器上有效） [cite: 6, 15, 183, 185]。
[[ICPC IN-CONTEXT PROMPT COMPRESSION WITH FASTER  INFERENCE]]
#### 7. PromptOptMe: Error-Aware Prompt Compression for LLM-based MT Evaluation Metrics (论文7)
- **面向LLM评估的、错误感知（Error-Aware）的输入数据压缩**：
    - **创新点**：PromptOptMe的核心思想是使用一个更小、经过微调的语言模型来压缩评估提示中的输入数据部分（即源文本和机器翻译文本）。关键在于这种压缩是“错误感知”的：在第一阶段的监督微调中，模型被训练来识别并保留源文本和MT文本中可能包含翻译错误的子串（error spans）。
    - **为什么提出/贡献**：LLM评估指标（如GEMBA-MQM）的准确性高度依赖于对原文和译文中错误细节的捕捉 。通用的提示压缩方法（如LLMLingua）可能无法针对性地保留这些对评估至关重要的细微错误信息 。PromptOptMe通过显式地训练压缩模型关注并保留这些错误片段，旨在确保压缩后的输入依然能让大型LLM做出准确的评估。这使得最先进的LLM评估指标（如GEMBA-MQM）更具成本效益和效率，增强了其广泛使用的可及性 。
- **创新的两阶段微调过程（监督微调 + 偏好优化）**：
    - **创新点**：论文为训练小型压缩模型设计了一个独特的两阶段流程 ：
        - **第一阶段：监督微调（Supervised Fine-Tuning）**：使用MQM（Multidimensional Quality Metrics）标注的数据，训练小型模型学习压缩任务的格式，包括生成压缩率、从源文和译文中提取潜在的错误片段，并生成确保这些错误片段完整的压缩文本 。
        - **第二阶段：偏好优化（Preference Optimization）**：使用ORPO（Odds-Ratio Preference Optimization）算法，根据实际评估指标（GEMBA-MQM）在压缩输入上的表现来进一步优化模型 。具体来说，模型会学习选择那些能使大型LLM（如GPT-4o）评估结果与使用未压缩文本时的评估结果差异最小的压缩版本 。
    - **为什么提出/贡献**：这种两阶段方法结合了监督学习对特定信息（错误片段）的强制保留能力和偏好学习对最终任务目标（评估分数一致性）的对齐能力。第一阶段确保了压缩的“底线”，即错误信息不丢失；第二阶段则从整体上优化压缩策略，使其更符合实际评估需求，从而在减少token的同时最大限度地保持评估质量。
- **针对MT评估场景的特化与显著的效率提升**：
    - **创新点**：该方法特别关注机器翻译（MT）评估场景，并以GEMBA-MQM这一先进的LLM评估指标作为应用和优化的起点 。实验结果显示，在不损失评估质量的情况下，token使用量减少了2.37倍（摘要中数据）或2.32倍（引言中数据）。
    - **为什么提出/贡献**：LLM在MT评估等任务中展现出高质量，但其高昂的token成本限制了其大规模应用 。PromptOptMe通过显著降低token使用量，使得这些先进的LLM评估指标在实际应用中（如在线重排序MT系统输出或处理网络规模数据集）变得更加经济可行 。
- **对评估提示中指令部分的简化（辅助贡献）**：
    - **创新点**：研究者还发现，GEMBA-MQM提示中冗长的指令部分（包含整个MQM错误类型学）可以被一个固定的、简化的指令模板所替代，而不会对评估质量产生负面影响 。
    - **为什么提出/贡献**：虽然这不是通过微调压缩模型实现的，但这一发现本身也为减少整体提示长度、降低计算成本做出了贡献，是对主要压缩方法的一个有益补充。
* **核心特点**：针对基于LLM的**机器翻译 (MT) 评估指标 (以GEMBA-MQM为起点)** [cite: 5, 212]，提出一种**错误感知 (error-aware)** 的提示压缩优化方法 [cite: 3, 212, 219]。
* **压缩模型**：使用一个更小的、经过微调的语言模型（如LLaMA-3.2）进行压缩 [cite: 3, 214, 219]。
* **两阶段微调**：
    * **监督微调 (SFT)**：训练模型学习压缩任务，并识别和保留源文本与MT文本中的潜在**错误跨度** (利用MQM标注数据) [cite: 4, 12, 67, 214, 219]。
    * **偏好优化 (使用ORPO算法)**：根据实际LLM评估指标（如GPT-40驱动的GEMBA-MQM）对不同压缩版本给出的分数差异，构建偏好数据，进一步细化压缩模型，使其选择能最好保持评估质量的压缩版本 [cite: 4, 13, 68, 214, 219]。
* **优势**：在不损失MT评估质量的情况下显著减少令牌使用量（高达2.37倍） [cite: 6, 34, 214]，使昂贵的LLM评估指标更具成本效益 [cite: 7, 214]。其针对性优化使其优于通用压缩方法 [cite: 162, 182, 183, 214]。
[[PromptOptMeError-Aware Prompt Compression for LLMbased MT Evaluation Metrics]]
#### 8. PIS: Linking Importance Sampling and Attention Mechanisms for Efficient Prompt Compression (论文8)
- **理论创新：基于度量理论的重要性采样与注意力机制的连接**：
    - **创新点**：论文为提示压缩建立了一个度量理论基础，将LLM注意力分数的分配形式化为可测函数，从而将token的重要性与注意力分布联系起来 。这为提示上下文优化提供了理论依据，不同于以往主要依赖启发式规则或外部模型的方法。
    - **为什么提出/贡献**：现有方法往往忽略LLM的内在机制 。通过将提示压缩问题置于重要性采样的理论框架下，并论证注意力分数可以作为token重要性的代理 ，PIS为如何选择和保留关键token提供了更根本的视角。这解决了“如何使提示压缩与LLM的计算机制对齐” 的研究问题。
- **PIS框架：双层级动态压缩机制**：
    - **创新点**：PIS采用了一个双层级的压缩机制，在token层面和句子层面进行重要性采样 。
        - **Token层面**：
            - 使用LLM原生的注意力分数（通过小型编码器模型高效近似提取 ）来量化token的显著性 。
            - 并非简单移除低注意力分数的token，而是优先考虑移除注意力分数_方差较高_的token，认为这些token更可能被LLM过度或不足采样 。
            - 引入TF-IDF分数作为注意力分数的修正，以防止重要的高注意力token被移除 。
            - 通过一个轻量级的9层强化学习网络（DDQN）来实现对每个句子压缩率的自适应调整 。
        - **句子层面**：
            - 提出了一种“俄罗斯轮盘赌”（Russian roulette）采样策略，根据句子间的相似性概率性地减少语义单元（句子）的冗余 。
    - **为什么提出/贡献**：这种双层级设计允许模型在不同粒度上优化提示。Token层面的细致处理（结合注意力分析、方差、TF-IDF和RL）旨在精确保留关键信息，而句子层面的采样则处理更大块的语义冗余。这种设计避免了完全依赖外部生成模型进行压缩 ，旨在实现更高的压缩质量和效率 。
- **高效压缩且不依赖外部生成模型进行压缩决策**：
    - **创新点**：PIS框架的设计避免了在压缩决策过程中依赖大型外部生成模型（如LLMLingua等方法所采用的）进行提示重写或评分 。它主要依赖LLM原生的注意力信息（通过小型编码器高效获取 ）和一个紧凑的强化学习模块 。
    - **为什么提出/贡献**：依赖外部大型模型进行压缩会引入显著的计算成本，增加了训练和推理的开销 。PIS通过利用LLM内部信号和小型RL网络，旨在实现一种既高效又有效的压缩方式。
- **潜在的推理效率提升**：
    - **创新点/观察**：论文指出，通过优化的上下文结构，PIS框架在一定程度上提高了下游任务的推理效率 。与原始输入相比，使用PIS压缩后的提示在下游任务上带来了5%的准确率提升 。
    - **为什么提出/贡献**：这表明了精心设计的提示压缩不仅能减少token数量，还可能通过改善上下文的组织方式，间接提升LLM的推理表现，这是一个额外的、积极的效应。
* **核心特点**：提出一种新颖的**提示重要性采样 (PIS)** 框架 [cite: 4, 228]，通过分析LLM原生的**注意力机制**的输出来采样重要token [cite: 4, 228]，从而动态压缩提示，并以**测度论**提供理论基础 [cite: 21, 29, 228, 234]。
* **双层压缩机制**：
    * **Token层面 (TIS)**：使用LLM（或小型编码器代理）的注意力得分（经TF-IDF校正）量化token显著性 [cite: 5, 106, 228, 234]。通过一个轻量级9层强化学习网络 (DDQN) 实现句子粒度的自适应压缩比率选择 [cite: 5, 112, 228, 234]。
    * **语义/句子层面 (SIS)**：在TIS处理后，基于句子间语义相似度（余弦相似度） [cite: 124, 228]，采用**俄罗斯轮盘赌采样策略**概率性移除冗余句子 [cite: 6, 123, 228, 234]。
* **优势**：无需外部生成模型 [cite: 22, 27, 235]，计算成本相对较低 [cite: 235]。在压缩质量和效率上均表现优越 [cite: 7, 25, 234]，有时优化的上下文结构甚至能提升下游任务性能 [cite: 8, 28, 234, 244]。
[[PIS Linking Importance Sampling and Attention Mechanisms for Efficient Prompt Compression]]
#### 9. Leveraging Attention to Effectively Compress Prompts for Long-Context LLMs (AttnComp) (论文9)
- **基于查询引导的交叉注意力（Query-Guided Cross-Attention）的Token重要性度量**：
    - **创新点**：针对现有方法（如基于信息熵 ）在评估token重要性时可能并非最优的问题 ，AttnComp提出使用语言模型内部的注意力机制，特别是“从查询到上下文的因果交叉注意力”（causal cross-attention from the query to the context），来评估每个token的重要性 。该方法首先识别小型因果语言模型中的“检索头”（retrieval heads） ，这些头在整合上下文信息时非常活跃 ，然后采用最大化策略（max strategy）整合这些头的注意力分数来得到最终的token重要性评分 。
    - **为什么提出/贡献**：与信息熵这类可能假设自然语言存在冗余但未必总是最优的经验性指标 相比，注意力机制，尤其是查询引导的交叉注意力，能够更精确地捕捉与特定查询相关的、细粒度的语义信息 。这为token重要性评估提供了一个更内在、更与LLM工作原理一致的度量方式，有望比基于熵的方法更好地保留关键信息。
- **基于自注意力（Self-Attention）和图算法的语义单元识别**：
    - **创新点**：为了解决传统压缩方法中独立评估和移除token可能破坏语义完整性的问题（即“独立性假设” ），AttnComp提出了一种将token聚类成“语义单元”（semantic units）的方法 。该方法假设token间的自注意力值能够有效捕捉它们之间的语义依赖程度 。具体步骤包括：
        1. 将文档中的token表示为一个全连接无向图的顶点，token间的自注意力分数（所有头的最大值）作为边的权重 。
        2. 构建该图的最大生成树（Maximum Spanning Tree, MST），以提取提示的核心语义结构 。
        3. 在MST上应用社区检测算法（如Louvain算法）来将token划分为不同的语义单元 。 压缩决策随后在这些语义单元的层面上进行，单元的重要性由其内部token的平均重要性分数决定 。
    - **为什么提出/贡献**：独立移除token可能导致上下文意义的丢失或改变 。通过将紧密相关的token（即自注意力值高的token对）组合成语义单元，并基于单元进行保留或移除决策，AttnComp旨在更好地维护压缩后提示的语义一致性和完整性，从而有效缓解独立性假设带来的问题 。
- **AttnComp整体框架的整合与有效性**：
    - **创新点**：AttnComp将上述两个核心思想（交叉注意力度量token重要性、自注意力识别语义单元）整合到一个统一的提示压缩框架中（如图1和算法1所示） 。它首先使用交叉注意力评估单个token的重要性，然后利用自注意力和图算法将token聚合成语义单元，并为这些单元分配重要性分数，最后根据压缩约束过滤掉重要性较低的语义单元 。
    - **为什么提出/贡献**：该框架提供了一种直接利用LLM内部注意力信号进行提示压缩的端到端方法。与依赖外部小型因果LM计算信息熵的方法相比，AttnComp旨在通过更深入地利用模型内部动态来取得更优的性能和更短的延迟 。实验结果表明，AttnComp在检索增强生成和多项长文本问答任务上超越了以往的基线方法 。
* **核心特点**：利用语言模型内部的**注意力机制**来指导提示压缩，解决传统信息熵指标的不足和token独立性假设问题 [cite: 8, 249, 252]。
* **重要性度量**：使用从**查询 (query) 到上下文 (context) 的因果交叉注意力 (CA)** (特别是从特定“检索头”提取) 作为评估token重要性的新指标 [cite: 9, 28, 29, 252, 253]。
* **语义单元识别**：开发了一种基于图的算法，利用上下文内部的**自注意力 (SA)** 构建图 [cite: 9, 34, 252]，并通过**最大生成树 (MST) 和社区检测 (Louvain算法)** 将token聚类成语义单元 [cite: 9, 34, 48, 49, 252]，在单元级别进行压缩决策 [cite: 252]。
* **优势**：提出的交叉注意力指标比PPL更优 [cite: 171, 172, 253, 259]。通过语义单元克服了独立性假设，更好地保留语义完整性 [cite: 35, 253, 261]。在性能和延迟方面均优于先前基线 [cite: 11, 12, 253, 260]。
[[Leveraging Attention to Effectively Compress Prompts for Long-Context LLMs]]

**关键技术细节**：
- **嵌入空间映射**：利用预训练模型的编码器，将整个Prompt或其结构化部分映射到高维嵌入向量。
- **低维投影技术**：应用主成分分析(PCA)、自动编码器(AE)、变分自动编码器(VAE)等技术将高维向量压缩到极低维度（如32-128维），保留主要语义信息。
- **Prompt重构方法**：
  - **向量解码重构**：使用专用解码器将压缩向量重构为自然语言Prompt
  - **语义引导生成**：使用压缩向量作为语义控制信号，引导LLM生成功能等效的新Prompt
  - **语义检索匹配**：在接收端维护语义索引库，搜索语义最接近的预存Prompt
- **量化与精度控制**：应用向量量化技术（如Product Quantization），进一步压缩传输比特数
可以向NLP领域中的
**优势**：
- 极高的压缩率，特别适合长文本Prompt
- 语义级保真度，关注功能等效而非完全复制
- 支持跨语言、跨模型的Prompt转换

**局限性**：
- 有损压缩，存在语义偏移风险
- 重构质量依赖于模型能力
- 计算开销较大，不适用资源受限场景
- 需预共享编解码器模型或参数

**适用场景**：适合要求功能等效但允许表述灵活的Prompt传递，特别是在行为信道带宽严格受限但通信双方计算资源较丰富的情况。在基于内容与行为双信道的隐蔽通信中，可用于传递全新任务指令或复杂控制逻辑。


