

> Previous neural network-based steganographic text generation models [13]–[17] are designed to generate words sequence of optimal conditional probabilities as much as possible.
> 以往基于神经网络的隐写文本生成模型 [cite: 13, 14, 15, 16, 17] 旨在尽可能生成最优条件概率的词序列。
📌 **分析：**
* 重申了现有神经网络模型的主要优化目标：追求每个词在给定上下文下的概率最大化，从而使生成的文本在局部上看起来最自然、最流畅。

---

> This optimization goal can only guarantee the generated steganographic text of high quality.
> 这种优化目标只能保证生成的隐写文本具有高质量。
📌 **分析：**
* 强调了现有方法的局限性：它们只侧重于文本的“质量”，即感知不可察觉性。这为后面提出“质量不等于安全性”做铺垫。

---

> But is the better quality of the generated steganographic sentences, the better imperceptibility of them?
> 但是，生成的隐写语句质量越好，其不可察觉性就越好吗？
📌 **分析：**
* 提出了一个直接的疑问，也是本节乃至本文的核心问题。它挑战了领域内普遍存在的直观假设。

---

> We did some experiments to verify this conjecture.
> 我们进行了一些实验来验证这个猜想。
📌 **分析：**
* 表明作者通过实证方法来回答上述质疑，体现了研究的严谨性。

---

> In order to illustrate our motivation and model design more conveniently, we will first show some experimental results in this section.
> 为了更方便地说明我们的动机和模型设计，我们将在本节中首先展示一些实验结果。
📌 **分析：**
* 解释了本节将实验结果提前展示的原因，即为了强化研究动机，让读者更直观地理解为何要提出 VAE-Stega。这种组织方式在研究论文中并不常见，但有助于突出核心发现。

---

> More details about the experiment settings and results can be found in Section V.
> 有关实验设置和结果的更多详细信息可在第五节中找到。
📌 **分析：**
* 补充说明，详细的实验方法和更多结果将在后续章节提供，避免本节内容过于冗长。

---

> Firstly, we trained the RNN-Stega model [13] on IMDB dataset [39], and then used the trained RNN-Stega model to generate 1,000 steganographic sentences at each different embedding rate.
> 首先，我们在 IMDB 数据集 [cite: 39] 上训练了 RNN-Stega 模型 [cite: 13]，然后使用训练好的 RNN-Stega 模型以不同的嵌入率生成了 1,000 条隐写语句。
📌 **分析：**
* **实验步骤1**：
    * **模型选择**：**RNN-Stega** [cite: 13] 作为基线模型，因为它代表了当时基于神经网络的先进文本生成隐写方法，其优化目标是生成高质量文本。
    * **数据集**：**IMDB 数据集** [cite: 39]，一个常用的、大规模的文本数据集，用于训练语言模型。
    * **生成数量和条件**：在不同**嵌入率**（embedding rate，即每词嵌入的比特数，后文简称 bpw）下生成 **1,000 条隐写语句**，以保证实验结果的统计显著性。

---

> Secondly, we calculated the mean perplexity of these generated steganographic sentences under different embedding rates.
> 其次，我们计算了这些生成的隐写语句在不同嵌入率下的**平均困惑度（mean perplexity）**。
📌 **分析：**
* **实验步骤2**：
    * **指标**：**平均困惑度（Perplexity）**，是衡量语言模型质量的标准客观指标。困惑度越低，表示模型对文本的预测能力越强，文本质量越好，通常意味着更好的语法和语义流畅性。
    * 困惑度在这里代表了**感知不可察觉性**的一个客观衡量。

---

> In the field of Natural Language Processing, perplexity is a standard measure for language model testing [40], it’s mathematical expression can be found in Equation (27).
> 在自然语言处理领域，困惑度是语言模型测试的标准度量 [cite: 40]，其数学表达式可在公式 (27) 中找到。
📌 **分析：**
* 进一步解释了困惑度的定义和在 NLP 领域的地位，增加了其作为评估指标的权威性。

---

> Usually, the smaller the value of perplexity, the better language model of the generated sentences.
> 通常，困惑度值越小，生成的语句的语言模型越好。
📌 **分析：**
* 重申了困惑度指标的含义，为后续分析结果做准备。

---

> Thirdly, we mixed the sentences with different embedding rates, then we strictly followed the requirements of the double-blind experiment and invited 11 people who had been screened by the English expression and reading ability to rate the quality of these sentences (1-5 points, the higher, the better).
> 第三，我们混合了不同嵌入率的句子，然后我们严格遵循**双盲实验（double-blind experiment）**的要求，邀请了11位经过英语表达和阅读能力筛选的人员来评价这些句子的质量（1-5分，分数越高越好）。
📌 **分析：**
* **实验步骤3**：
    * **双盲实验**：实验设计中的一种方法，旨在消除主观偏见。在这种实验中，参与者（这里是评分人员）和实验执行者（这里是组织者，但不直接接触评分人员）都不知道哪些是隐写文本，哪些是正常文本，也不知道它们对应的嵌入率，从而确保评价的客观性。
    * **人工评分**：直接的人类主观评估，用于衡量文本的**感知不可察觉性**。这是非常重要的，因为最终的“不可察觉”是针对人类观察者而言的。
    * **评分标准**：1-5分，分数越高表示质量越好。

---

> We calculated the average human scores of these generated steganographic sentences under different embedding rates, which can represent the subjective quality of these generated sentences.
> 我们计算了这些生成的隐写语句在不同嵌入率下的平均人工评分，这可以代表这些生成语句的**主观质量（subjective quality）**。
📌 **分析：**
* 进一步明确了人工评分的目的和意义：衡量主观质量，作为感知不可察觉性的另一个重要指标。

---

> Finally, we mixed steganographic sentences at each different embedding rate with equal amount of normal sentences, and then used the recently proposed text steganalysis model [41] to detect them.
> 最后，我们将每种不同嵌入率下的隐写语句与等量的正常语句混合，然后使用最近提出的文本隐写分析模型 [cite: 41] 来检测它们。
📌 **分析：**
* **实验步骤4**：
    * **隐写分析检测**：这是直接衡量**统计不可察觉性**的关键步骤。如果隐写分析模型能够区分隐写文本和正常文本，就说明隐写方法在统计上是可察觉的，安全性较低。
    * **混合比例**：隐写语句与正常语句“等量混合”，模拟了真实的检测场景，使得检测任务具有二分类的性质。
    * **隐写分析模型**：使用当时最先进的隐写分析工具 [cite: 41]，以确保评估结果具有说服力。

---

> We repeated this experiment 10 times and recorded the mean and standard deviation of the detection accuracy (calculated as formula (30)) under different embedding rates.
> 我们将此实验重复了10次，并记录了在不同嵌入率下的检测准确率的平均值和标准差（根据公式 (30) 计算）。
📌 **分析：**
* **重复实验和统计量**：重复10次实验并记录平均值和标准差，是为了提高实验结果的可靠性和统计显著性，减少随机性对结果的影响。
* **检测准确率**：这是衡量隐写方法安全性的核心指标之一，准确率越低，说明隐写方法越安全。

---

> The experiment results¹ are shown in Figure 2, in which the abscissa represents the average number of bits embedded in per word (bpw).
> 实验结果¹ 如图2所示，其中横坐标表示每词平均嵌入的比特数（bpw）。
📌 **分析：**
* 引导读者查看图2，并解释了图2的横坐标含义。

---

> The orange line indicates the calculated mean perplexity (the smaller, the better) of these generated steganographic sentences.
> 橙色线表示这些生成的隐写语句的计算平均困惑度（值越小越好）。
📌 **分析：**
* 解释图2中橙色线代表的含义，再次强调困惑度是“越小越好”。

---

> The size and transparency of the dots represent the average human score (the larger, the better), with the specific score at the top.
> 圆点的大小和透明度代表平均人工评分（值越大越好），具体分数显示在顶部。
📌 **分析：**
* 解释图2中圆点代表的含义，强调人工评分是“越大越好”。这两种指标（困惑度、人工评分）共同衡量**感知不可察觉性**。

---

> The blue line represents the steganographic detection accuracy (the lower, the better) under different embedding rates.
> 蓝色线表示在不同嵌入率下的隐写分析检测准确率（值越低越好）。
📌 **分析：**
* 解释图2中蓝色线代表的含义，强调检测准确率是“越低越好”，这衡量了**统计不可察觉性**。通过图2，读者可以直观地看到困惑度/人工评分（质量）与检测准确率（安全）之间的关系。

---

> According to Figure 2, we can draw the following conclusions.
> 根据图2，我们可以得出以下结论。
📌 **分析：**
* 引导读者对图2进行分析和总结。
![[Pasted image 20250521143339.png]]
---

> Firstly, with the increase of the embedding rate, the trend of perplexity and the human score of the generated steganographic sentences is consistent, they both indicate that the quality of the generated steganographic sentences are getting worse.
> 首先，随着嵌入率的增加，生成的隐写语句的困惑度和人工评分的趋势是一致的，它们都表明生成的隐写语句质量正在变差。
📌 **分析：**
* **结论1（符合预期）**：高嵌入率意味着要隐藏更多的信息，这必然会迫使生成模型做出更多“不那么自然”的选择，从而导致文本质量下降。这符合常识和以往研究的结论。

---

> This is in line with our expectations and also the conclusions of previous related works [13], [16], [17].
> 这符合我们的预期，也符合以往相关工作的结论 [cite: 13, 16, 17]。
📌 **分析：**
* 强调了第一个结论的普遍性和可信度。

---

> Secondly, we found that the quality of the generated steganographic sentences is not equivalent to the imperceptibility of them.
> 其次，我们发现生成的隐写语句的质量与它们的不可察觉性并不等同。
📌 **分析：**
* **结论2（核心发现）**：直接回答了本节开头提出的问题：“质量越好，不可察觉性就越好吗？”答案是“不等同”。这正式引出了 Psic Effect。

---

> As shown in Figure 2, when bpw is less than 2, the quality of the generated steganographic sentences are the best, but they are the easiest to be detected; when bpw is around 5, the quality of the generated steganographic sentences are the worst, but its ability to resist steganalysis is the strongest.
> 如图2所示，当 bpw 小于2时，生成的隐写语句质量最好，但它们最容易被检测到；当 bpw 约为5时，生成的隐写语句质量最差，但其抵抗隐写分析的能力最强。
📌 **分析：**
* **具体证据**：通过引用图2中的具体数据点，直观地展示了质量（困惑度/人工评分）和安全性（检测准确率）之间的反常关系，完美诠释了 Psic Effect。
    * **低bpw**：质量高 -> 容易被检测
    * **高bpw**：质量低 -> 不易被检测

---

> Thirdly, we noticed a very unusual phenomenon: with the increase of embedding rate, the accuracy of steganographic detection gradually decreases, which means that the generated steganographic sentences become more difficult to detect.
> 第三，我们注意到一个非常不寻常的现象：随着嵌入率的增加，隐写检测的准确率逐渐降低，这意味着生成的隐写语句变得更难以检测。
📌 **分析：**
* **核心现象（再次强调）**：重申了“嵌入率越高，检测准确率反而越低”这一反直觉的现象，进一步强化了 Psic Effect 的存在。

---

> This phenomenon is not isolated. In the experimental part of Section V, we used a variety of steganalysis algorithms to verify this phenomenon from multiple datasets and finally draw the same conclusion.
> 这种现象并非孤立存在。在第五节的实验部分，我们使用了多种隐写分析算法，并从多个数据集验证了这一现象，最终得出了相同的结论。
📌 **分析：**
* **普遍性验证**：强调了这种现象的普遍性，不仅仅是针对 RNN-Stega 模型或特定数据集的偶然结果，而是在更广泛的场景下都存在的。这增强了 Psic Effect 的可靠性和重要性。

---

> Ziegler et al. [17] also reported similar experimental results in a recent preprint.
> 齐格勒 等人 [cite: 17] 在最近的一篇预印本中也报告了类似的实验结果。
📌 **分析：**
* 提供了外部证据支持：其他研究者 [cite: 17] 也观察到了类似的现象，这进一步证实了 Psic Effect 的存在性和普遍性，而非本文作者的独有发现。

---

> Their experiments showed that as the embedding rate increases, the KL divergence of the conditional probability distribution of the generated steganographic sentences and that of normal sentences decreased gradually (the lower, the better), but the human evaluation score also decreased gradually.
> 他们的实验表明，随着嵌入率的增加，生成的隐写语句的条件概率分布与正常语句的条件概率分布之间的KL散度逐渐减小（值越小越好），但人工评价得分也逐渐降低。
📌 **分析：**
* **具体数据对比**：引用 Ziegler et al. [cite: 17] 的研究，指出了他们的发现与本文的 Psic Effect 相符：
    * **KL散度（条件概率分布）降低**：说明在嵌入率高时，生成的文本在“局部”统计特性上与正常文本更接近。
    * **人工评价得分降低**：说明文本质量下降。
* 这与本文图2的趋势一致，进一步说明了“局部统计相似性增加（KLD降低）”与“感知质量下降（人工得分降低）”之间的关系。

---

> This phenomenon seems to be different from our usual cognition.
> 这种现象似乎与我们通常的认知不同。
📌 **分析：**
* 再次强调了 Psic Effect 的反直觉性，即它与我们通常认为“信息嵌入越多，越容易被发现”的观念相悖。这正是需要深入研究和解决的问题。

---

> In general, we believe that as the amount of additional information embedded in a carrier increases, the imperceptibility of the carrier will decrease, which will be reflected in the increase of steganalysis accuracy [5].
> 一般来说，我们认为随着载体中嵌入的额外信息量增加，载体的不可察觉性将降低，这将在隐写分析准确率的增加中体现出来 [cite: 5]。
📌 **分析：**
* 明确指出了传统认知：高嵌入量导致低隐蔽性，隐写分析准确率升高。通过对比这种“一般认知”与实际观测到的“Psic Effect”，突显了 Psic Effect 的异常性和研究价值。引用 [5] 提供了这种传统认知的文献支持。


> In order to further explain this anomalous phenomenon, we carry out more in-depth experimental analysis.
> 为了进一步解释这种异常现象，我们进行了更深入的实验分析。
📌 **分析：**
* 承接上文发现的“Psic Effect”，本句表明作者将通过更详细的实验来探究其背后的原因。

---

> We calculated the perplexity of each generated steganographic sentence under different embedding rates and compared their distribution with that of normal sentences.
> 我们计算了在不同嵌入率下每个生成的隐写语句的困惑度，并将其分布与正常语句的困惑度分布进行了比较。
📌 **分析：**
* **实验方法**：不再仅仅计算平均困惑度，而是深入到“每个句子”的困惑度，并分析其**分布**。这是更精细的统计分析，旨在揭示宏观现象背后的微观原因。

---

> The results are shown in Figure 3, the black line represents the distribution of normal sentences, while the other color lines represent the steganographic text with different embedding rates.
> 结果如图3所示，黑线代表正常语句的分布，而其他彩线代表不同嵌入率的隐写文本。
📌 **分析：**
* 引导读者查看图3，并解释了图3中不同线条的含义。图3将是解释 Psic Effect 现象的关键图示。

---

> Figure 3 can well explain the phenomenon in Figure 2.
> 图3可以很好地解释图2中的现象。
📌 **分析：**
* 直接指出图3是理解 Psic Effect（图2所示现象）的关键。这预示着对图3的分析将揭示现象的内在机制。

---

> Texts on public social networks are written by people of different ages and backgrounds in different ways of expression.
> 公共社交网络上的文本由不同年龄、不同背景的人以不同的表达方式书写。
📌 **分析：**
* 指出了正常（真实）文本的特点：**多样性**和**非规范性**。这种多样性是理解其统计分布的关键。

---

> This leads to the fact that most of the human-written sentences may not obey the optimal language model and form a large variance.
> 这导致大多数人类书写的句子可能不遵循**最优语言模型（optimal language model）**并形成较大的**方差（variance）**。
📌 **分析：**
* **“最优语言模型”**：指理论上最符合语法和语义规则，最流畅自然的语言模式。
* **“较大方差”**：意味着人类写作的困惑度分布广泛，有非常自然流畅的句子，也有相对口语化、不那么“完美”的句子。
* 本句揭示了**正常文本的特性**：它们并非都完美符合“最优”模型，而是存在很大的多样性。这是理解为何“过度优化”生成的文本反而容易被发现的关键。

---

> The previous automatic text generation steganographic model [13], [16], [17] mainly focused on optimizing the language model and the encoding method of each word’s conditional probability distribution, that is $p(d_j | x_1, x_2, . . . , x_{t−1}, d_j ∈ CP)$, where CP indicates the candidate pool of each generated word, which is constructed by calculating the conditional probability of the words in the dictionary at each time and then selecting the words with high probability.
> 以往的自动文本生成隐写模型 [cite: 13, 16, 17] 主要侧重于优化语言模型和每个词的条件概率分布的编码方法，即 $p(d_j | x_1, x_2, . . . , x_{t−1}, d_j ∈ CP)$，其中 $CP$ 表示每个生成词的**候选池（candidate pool）**，它是通过在每个时间步计算词典中词的条件概率并选择高概率的词来构建的。
📌 **分析：**
* **“候选池（CP）”**：指在文本生成过程中，模型根据上下文预测出的可能作为下一个词的备选词集合，通常只包含概率较高的词。
* 再次重申了现有方法的优化目标和工作原理：它们致力于生成在局部上（通过条件概率和候选池选择）尽可能“好”的文本。

---

> They mainly aimed at solving the first challenge, that is, to make each generated steganographic sentence look fluent enough.
> 它们主要旨在解决第一个挑战，即让每个生成的隐写语句看起来足够流畅。
📌 **分析：**
* 强调了现有方法的核心目标仍然是“表面流畅”，也就是**感知不可察觉性**。

---

> When the embedding rate is small, such as bpw = 1, they can always choose words with almost the highest conditional probability as the output of each time step, so the quality of the steganographic sentences they generated can be good enough.
> 当嵌入率较小时，例如 bpw = 1，它们总能选择几乎具有最高条件概率的词作为每一步的输出，因此它们生成的隐写语句的质量可以足够好。
📌 **分析：**
* 解释了为什么在低嵌入率下文本质量好：秘密信息量小，模型有足够的自由度去选择最自然的词。

---

> Reflected in Figure 3, the mean and variance of their perplexity are small, forming a sharp spike on the left.
> 反映在图3中，它们的困惑度的均值和方差都很小，在左侧形成一个**尖锐的峰值（sharp spike）**。
📌 **分析：**
* **关键解释**：这句是理解 Psic Effect 机制的核心。
    * **“困惑度的均值和方差都很小”**：意味着在低嵌入率下，模型倾向于生成非常“完美”、高度符合语言模型预测的句子。
    * **“尖锐的峰值在左侧”**：在困惑度分布图上，这表现为分布非常集中在低困惑度区域（高质量），且分布范围狭窄（方差小）。
* 这种“完美”与真实世界文本的“多样性”形成了对比，正是隐写分析能够利用的线索。

---

> However, this also cause a huge difference between the overall distribution of generated steganographic sentences and that of normal sentences, which makes them to be easily distinguished.
> 然而，这也导致了生成的隐写语句的整体分布与正常语句的整体分布之间存在巨大差异，这使得它们容易被区分。
📌 **分析：**
* **Psic Effect 的解释**：这就是低嵌入率下文本质量高却易被检测的原因。当生成文本的困惑度分布过于“完美”和“集中”时，它与真实、多样的正常文本的**整体分布**形成了鲜明对比，隐写分析器就能轻易地识别出这种“不自然”的“完美”。

---

> As the embedding rate increases, the candidate pool for each word gradually increases, and the CP gradually approaches to the entire dictionary D, so the conditional probability distribution of each word in the generated steganographic text gradually approaches to that of normal sentences, so it becomes more and more difficult to distinguish statistically.
> 随着嵌入率的增加，每个词的**候选池（candidate pool）**逐渐增大，并且 $CP$ 逐渐接近整个词典 $D$，因此生成的隐写文本中每个词的条件概率分布逐渐接近正常语句的条件概率分布，所以统计上越来越难以区分。
📌 **分析：**
* **高嵌入率的解释**：
    * **“候选池逐渐增大”**：为了嵌入更多的信息，模型被迫从更大的候选词集合中选择词，甚至包括那些条件概率较低的词。
    * **“CP 逐渐接近整个词典 D”**：这意味着模型在选择词时，考虑的范围更广，不再局限于“最优”的词。
    * **“条件概率分布逐渐接近正常语句的条件概率分布”**：当模型从更广泛的词集中选择时，其选择行为可能变得更“随机”，从而在局部统计特性上更接近正常语言的自然随机性，隐写分析更难发现。
* 这解释了为什么在**高嵌入率**下，尽管文本质量下降，但反而更难被统计检测。因为它的行为变得不那么“刻意优化”，反而更像真实的、有噪声的自然语言。

---

> But at the same time, the model is more and more likely to select words with lower conditional probability, so the text quality begins to decline, reflected in the mean value of the perplexity distribution curve in Figure 3 gradually shifting to the right, and the variance gradually increasing.
> 但同时，模型越来越可能选择条件概率较低的词，因此文本质量开始下降，这反映在图3中困惑度分布曲线的平均值逐渐向右移动，方差也逐渐增大。
📌 **分析：**
* **质量下降的解释**：在高嵌入率下，由于模型被迫选择概率较低的词，必然导致生成的文本不再那么流畅和自然，即质量下降。
* **图3的体现**：
    * **“平均值逐渐向右移动”**：困惑度增加，质量变差。
    * **“方差也逐渐增大”**：困惑度分布更宽泛，更接近真实人类写作的多样性，不再是尖锐的“完美峰值”。

---

> With the increase of embedding rate, the different trend of the quality and anti-steganaly sis ability of generated steganographic carriers represent two different aspects of these generated steganographic carriers, we name them perceptual-imperceptibility and statistical-imperceptibility, which has been shown in Figure 4.
> 随着嵌入率的增加，生成的隐写载体的质量和抗隐写分析能力的不同趋势代表了这些生成隐写载体的两个不同方面，我们将其命名为**感知不可察觉性（perceptual-imperceptibility）**和**统计不可察觉性（statistical-imperceptibility）**，如图4所示。
📌 **分析：**
* 总结了 Psic Effect 现象，并正式引入了两个核心概念：**感知不可察觉性**（与质量相关）和**统计不可察觉性**（与抗隐写分析能力相关）。图4应是这两个概念的示意图。
![[Pasted image 20250521144452.png]]
---

> Perceptual-imperceptibility measures the quality of the generated steganographic carriers, while statistical-imperceptibility measures the statistical inseparability between the generated steganographic carriers and normal carriers.
> 感知不可察觉性衡量生成的隐写载体的质量，而统计不可察觉性衡量生成的隐写载体与正常载体之间的**统计不可分离性（statistical inseparability）**。
📌 **分析：**
* 对这两个核心概念的精确定义：
    * **感知不可察婷性**：关注的是“看得像不像”（human perception）。
    * **统计不可察觉性**：关注的是“数据是不是有统计上的异常”（statistical properties）。
* **“统计不可分离性”**：指在统计学上无法将隐写载体与正常载体区分开来。这是隐写术追求的最高安全目标。

---

> We think that the unique phenomenon showed in Figure 2 is not accidental, it may be a common paradigm for generative steganography methods.
> 我们认为图2中显示的独特现象并非偶然，它可能是生成式隐写方法的一个**共同范例（common paradigm）**。
📌 **分析：**
* 强调了 Psic Effect 的普遍性：这不仅仅是 RNN-Stega 的问题，而是所有基于“生成”而非“修改”的隐写方法可能都会面临的挑战。

---

> Because the feature expression of an information carrier is almost infinite, the limited fitting ability of artificially designed models can hardly cover all explicit or implicit feature expressions.
> 因为信息载体的特征表达几乎是无限的，人工设计的模型的有限拟合能力很难覆盖所有显式或隐式特征表达。
📌 **分析：**
* 解释 Psic Effect 普遍性的深层原因：自然语言的复杂性和多样性是无限的。
* **“有限拟合能力”**：指即使是复杂的神经网络模型，其学习能力也受限于模型结构、数据量和训练目标。
* **“显式或隐式特征表达”**：指语言中可见的语法、语义，以及更深层次的、难以捕捉的统计模式和风格特征。
* 这句话暗示了，当模型过度优化某个显式特征（如语言模型）时，可能会无意中破坏了其他隐式特征的自然性。

---

> The “extreme optimization” of a few features (such as the language model of sentences) may harm the overall statistical distribution pattern of the carriers, as shown in Figure 2 and Figure 3.
> 对少数特征（例如句子的语言模型）的“**极致优化（extreme optimization）**”可能会损害载体的**整体统计分布模式（overall statistical distribution pattern）**，如图2 [cite: 3] 和图3 [cite: 3] 所示。
📌 **分析：**
* **“极致优化”**：这是 Psic Effect 产生的根本原因。当模型只顾着把文本生成得“太完美”（在某个局部指标上），它就可能脱离真实世界文本的整体分布，反而变得容易被发现。
* **“整体统计分布模式”**：指文本在宏观层面的统计规律，例如词频分布、句子长度分布、句法复杂度分布等。

---

> For generative steganography, as the generative model continues to be optimized to a certain stage, the perceptual-imperceptibility and statistical-imperceptibility may display conflict patterns, which we call the Perceptual-Statistical Imperceptibility Conflict Effect (Psic Effect).
> 对于生成式隐写术，当生成模型持续优化到一定阶段时，感知不可察觉性和统计不可察觉性可能会呈现冲突模式，我们称之为**感知-统计不可察觉性冲突效应（Psic Effect）**。
📌 **分析：**
* 正式对“Psic Effect”进行了总结性定义，强调了其在生成式隐写术中的普遍性和重要性。

---

> Facing the Psic Effect, Alice should balance these two aspects of imperceptibility.
> 面对 Psic 效应，爱丽丝（Alice）应该平衡不可察觉性的这两个方面。
📌 **分析：**
* 提出了应对 Psic Effect 的策略：不能只顾一边，必须同时兼顾感知不可察觉性和统计不可察觉性。这是本文方法（VAE-Stega）设计的核心指导思想。

---

> If Alice only considers perceptual imperceptibility, she can generate steganographic carriers with high quality, but it may make it easily be detected by Eve through statistical analysis due to the large differences in overall distribution with normal carriers.
> 如果爱丽丝（Alice）只考虑感知不可察觉性，她可以生成高质量的隐写载体，但由于与正常载体在整体分布上存在巨大差异，夏娃（Eve）可能通过统计分析轻易检测到它们。
📌 **分析：**
* 再次强调了只关注“质量”的弊端，即导致统计上的不隐蔽。

---

> If Alice only considers statistical-imperceptibility, the quality of the generated steganographic carriers may be very poor, which may also lead to being easily recognized by human Eve.
> 如果爱丽丝（Alice）只考虑统计不可察觉性，生成的隐写载体质量可能非常差，这同样可能导致被人类夏娃（Eve）轻易识别。
📌 **分析：**
* 强调了只关注“统计隐蔽性”的弊端：如果文本本身支离破碎、语法不通，即使统计上接近正常文本，也会被人类一眼识破。

---

> In the rest of this paper, we will continue to take the text as the study subject and illustrate how to use the VAE architecture to balance and optimize the two aspects of imperceptibility under the influence of the Psic Effect.
> 在本文的其余部分，我们将继续以文本为研究对象，说明如何在 Psic 效应的影响下，使用 **VAE 架构（VAE architecture）**来平衡和优化不可察觉性的两个方面。
📌 **分析：**
* 总结了本节引出的问题，并再次预告了即将介绍的 VAE-Stega 模型，它正是为了解决这一平衡问题而设计的。
* **“VAE 架构”**：指变分自编码器这一模型结构。
* 再次强调了本文的核心任务和技术路线。