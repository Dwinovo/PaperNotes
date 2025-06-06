
> For a text set with a dictionary D, each sentence S with length n is sampled out from the space $C = \{w_i |w_i ∈ D\}^n$, and the total sentences in space C will form a probability distribution $P_C$.
> 对于一个拥有字典 $D$ 的文本集，每个长度为 $n$ 的句子 $S$ 都从空间 $C = \{w_i |w_i ∈ D\}^n$ 中采样，并且空间 $C$ 中的所有句子将形成一个概率分布 $P_C$。
📌 **分析：**
* **词典 $D$**：包含所有可能词汇的集合。
* **空间 $C$**：表示所有可能的长度为 $n$ 的句子组合。
* **概率分布 $P_C$**：这是正常文本（cover text）的真实分布。隐写术的目标就是让生成的隐写文本的分布尽可能接近 $P_C$。

---

> According to Kerckhoff’s principle [42], we can assume that Eve has a complete knowledge of the steganographic channel, including the used cover text set and its distribution $P_C$, as well as the steganographic method that Alice may choose.
> 根据**克尔克霍夫原则（Kerckhoff’s principle）** [cite: 42]，我们可以假设夏娃（Eve）完全了解隐写信道，包括所使用的原始文本集及其分布 $P_C$，以及爱丽丝（Alice）可能选择的隐写方法。
📌 **分析：**
* **“克尔克霍夫原则”** [cite: 42]：密码学中的一条基本原则，它指出一个密码系统的安全性不应依赖于其设计的保密性，而应依赖于其密钥的保密性。在隐写术中，这意味着隐写算法本身是公开的，Eve知道 Alice 可能使用的所有技术细节，除了秘密密钥。
* **假设 Eve 拥有完整知识**：这是一个在信息安全领域进行理论分析和模型设计时的强假设。这意味着如果你的隐写方法在这种“最坏情况”下仍然安全，那么在实际中它也可能更安全。

---

> Therefore, Eve can construct a large number of steganographic carriers using the corresponding steganographic algorithm, so we can assume that Eve also knows the statistical distribution of steganographic carrier space, which can be descripted as $P_S$.
> 因此，夏娃（Eve）可以使用相应的隐写算法构建大量隐写载体，所以我们可以假设夏娃也知道隐写载体空间的统计分布，可以描述为 $P_S$。
📌 **分析：**
* 承接克尔克霍夫原则，由于 Eve 知道算法，她就可以生成大量的隐写样本，从而推断出隐写载体（stego text）的统计分布 $P_S$。

---

> The information-theoretic definition of the security is quantified in terms of the Kullback-Leibler (KL) divergence [34] of observation $x \in X$ between the distributions of C and S: $D_{KL} (P_C ||P_S ) = \sum_{x∈X} P_C(x )log \frac{P_C(x )}{P_S (x )}$. (3)
> 安全性的信息论定义是根据原始载体 $C$ 和隐写载体 $S$ 的分布之间，观察值 $x \in \mathcal{X}$ 的**库尔巴克-莱布勒散度（Kullback-Leibler (KL) divergence）** [cite: 34] 来量化的：
> $$D_{KL} (P_C ||P_S ) = \sum_{x \in \mathcal{X}} P_C(x ) \log \frac{P_C(x )}{P_S (x )} \quad \text{(3)}$$
📌 **分析：**
* **“信息论定义”**：这是隐写安全性的一个重要理论基础。
* **“Kullback-Leibler (KL) 散度”** [cite: 34]：衡量两个概率分布之间差异的非对称指标。
    * **$P_C(x)$**：正常文本中观察值 $x$ 出现的概率。
    * **$P_S(x)$**：隐写文本中观察值 $x$ 出现的概率。
    * KL 散度越小，表示两个分布越接近，隐写效果越好，隐蔽性越高。如果 $P_C$ 和 $P_S$ 完全相同，则 KL 散度为 0。

---

> In order to ensure the imperceptibility of the whole steganographic system and resist steganalysis from Eve, the core goal of steganographic methods is to reduce the statistical distribution difference between cover carriers and steganographic carriers, that is, to reduce the KL divergence of $P_S$ and $P_C$, which is $D_{KL} (P_C||P_S )$.
> 为了确保整个隐写系统的不可察觉性并抵抗夏娃（Eve）的隐写分析，隐写方法的核心目标是减少原始载体和隐写载体之间的统计分布差异，即将 $P_S$ 和 $P_C$ 的 KL 散度 $D_{KL} (P_C||P_S )$ 降到最低。
📌 **分析：**
* 明确了隐写方法的核心优化目标：最小化正常载体和隐写载体之间的 KL 散度，从而在统计上实现不可区分性。这直接对应了“统计不可察觉性”。

---

> In order to achieve this goal, in this paper, we propose a text generation steganography based on variational auto-encoder (VAE-Stega).
> 为了实现这一目标，在本文中，我们提出了一种基于**变分自编码器（variational auto-encoder, VAE）**的文本生成隐写术（**VAE-Stega**）。
📌 **分析：**
* 再次重申了本文提出的 VAE-Stega 模型，并将其定位为解决上述目标（最小化 KL 散度）的方案。

---

> Compared with the previous methods, VAE-Stega further considers the overall statistical distribution characteristic of normal sentences in the process of generating steganographic sentences.
> 与以往的方法相比，VAE-Stega 在生成隐写语句的过程中进一步考虑了正常语句的**整体统计分布特征（overall statistical distribution characteristic）**。
📌 **分析：**
* **VAE-Stega 的核心优势**：强调了 VAE-Stega 与现有方法的关键区别。现有方法主要关注局部语言模型（如条件概率），而 VAE-Stega 更进一步，考虑了“整体”的统计分布特征，这正是解决 Psic Effect 的关键。

---

> Thus, it can further improve statistical-imperceptibility while ensuring certain perceptual-imperceptibility.
> 因此，它可以在保证一定感知不可察觉性的同时，进一步提高统计不可察觉性。
📌 **分析：**
* 总结了 VAE-Stega 的核心贡献：在牺牲少量感知质量的情况下，显著提升统计不可察觉性，从而实现两者之间的平衡。

---

> A. Variational Auto-Encoder Structure
> A. **变分自编码器结构**
📌 **分析：**
* 本小节将详细介绍 VAE 的基本原理和结构，以及它如何应用于 VAE-Stega。

---

> Variational auto-encoder is a generative model which was proposed by Kingma and Welling [43] and Rezende et al. [44], it has shown amazing performance on a variety of media generation tasks [45]–[47].
> **变分自编码器（Variational auto-encoder）**是由 Kingma 和 Welling [cite: 43] 以及 Rezende 等人 [cite: 44] 提出的**生成模型（generative model）**，它在各种媒体生成任务中都表现出惊人的性能 [cite: 45, 46, 47]。
📌 **分析：**
* **“生成模型”**：指能够学习数据分布并生成新数据的模型（与判别模型相对，判别模型旨在区分不同类别）。
* 引入了 VAE 的背景，并强调了其在生成任务中的强大能力，为在文本生成隐写中的应用提供了依据。

---

> A variational auto-encoder conforms to the general auto-encoder architecture, which typically maps a single sample x to a feature space $z ∈ Z$ through an encoder and then reconstructs the feature into a raw sample using a decoder, that is: { Encoder : X → Z, f (x) = z, Decoder : Z → X , g(z) = x . (4)
> 变分自编码器遵循一般的自编码器架构，通常通过**编码器（encoder）**将单个样本 $x$ 映射到**特征空间（feature space）** $z ∈ \mathcal{Z}$，然后使用**解码器（decoder）**将该特征重构回原始样本 $x'$，即：
> $$\begin{cases} \text{Encoder} : \mathcal{X} \rightarrow \mathcal{Z}, & f(x) = z, \\ \text{Decoder} : \mathcal{Z} \rightarrow \mathcal{X}, & g(z) = x'. \end{cases} \quad \text{(4)}$$
📌 **分析：**
* **自编码器 (Auto-encoder)**：一种无监督学习模型，旨在通过学习输入数据的压缩表示（编码），然后从该表示中重构数据（解码）。
* **编码器 ($f(x)=z$)**：将原始数据 $x$ 压缩到潜在空间 $Z$ 中的特征表示 $z$。
* **解码器 ($g(z)=x'$)**：将潜在表示 $z$ 重构回原始数据空间中的 $x'$。
* 公式 (4) 简洁地表示了自编码器的基本结构。

---

> Then by minimizing the difference between the reconstructed sample x and the original sample x, we can get the optimal mapping from the sample space to the feature space.
> 然后通过最小化重构样本 $x'$ 和原始样本 $x$ 之间的差异，我们可以得到从样本空间到特征空间的最佳映射。
📌 **分析：**
* 解释了自编码器的训练目标：通过最小化重建误差（即 $x$ 和 $x'$ 之间的差异），迫使编码器学习到有意义的、能够代表原始数据的特征。

---

> However, an auto-encoder only learns the point-to-point mapping between samples and features, so the decoder can only reconstruct the original samples, but can not generate new samples.
> 然而，自编码器只学习样本和特征之间的**点对点映射（point-to-point mapping）**，因此解码器只能重构原始样本，而不能生成新样本。
📌 **分析：**
* **“点对点映射”**：这是普通自编码器的局限性。它学习的是输入到输出的确定性映射，其潜在空间通常是离散的或不连续的，无法在潜在空间中随机采样来生成新的、未见过的但符合原始数据分布的样本。这是 VAE 出现的原因。

---

> Since the statistical-imperceptibility requires that the statistics distribution of generated steganographic sentences and that of normal sentences should be close enough, then our basic idea is to find an encoding function to embed normal sentences to a latent space Z, forming a distribution of $p_Z$.
> 由于统计不可察觉性要求生成的隐写语句的统计分布与正常语句的统计分布足够接近，那么我们的基本思想是找到一个编码函数，将正常语句嵌入到**潜在空间（latent space）** $Z$ 中，形成一个分布 $p_Z$。
📌 **分析：**
* **“潜在空间 Z”**：指数据经过编码器压缩后的低维表示空间。
* **核心思想**：为了实现统计不可察觉性，本文的目标是让生成文本的统计分布与正常文本的统计分布在潜在空间中保持一致。这通过 VAE 来实现。

---

> After that, we can randomly sample from the latent space according to its distribution and get a sampled latent vector z.
> 之后，我们可以根据其分布从潜在空间中随机采样得到一个采样后的潜在向量 $z$。
📌 **分析：**
* 强调了 VAE 的生成能力：一旦学习到了潜在空间的分布，就可以从中随机采样，从而生成新的、未见过的样本。这是普通自编码器无法做到的。

---

> We send this vector z into the decoder and then generate a sentence under the constraint of z, thus can keep the distribution of generated sentences conforms to that of normal sentences.
> 我们将这个向量 $z$ 送入解码器，然后在 $z$ 的约束下生成一个句子，这样就可以使生成句子的分布符合正常句子的分布。
📌 **分析：**
* 解释了 VAE 在生成隐写文本中的作用：通过从符合正常文本分布的潜在空间中采样 $z$，并以此 $z$ 作为解码器的输入约束，可以确保生成的隐写文本的整体统计分布与正常文本相似。

---

> The process of decoder can be expressed as $g_{\theta} (x|z)$. It defines a joint probability distribution over data and latent variables: $p_{\theta} (x, z)$.
> 解码器的过程可以表示为 $g_{\theta} (x|z)$。它定义了数据和潜在变量的**联合概率分布（joint probability distribution）**：$p_{\theta} (x, z)$。
📌 **分析：**
* **$g_{\theta} (x|z)$**：解码器的参数化形式，表示给定潜在变量 $z$ 时，生成数据 $x$ 的条件概率。
* **$p_{\theta} (x, z)$**：联合概率分布，表示数据 $x$ 和潜在变量 $z$ 同时出现的概率。

---

> We can decompose this into the likelihood and prior: $p_{\theta} (x , z) = p_{\theta} (x |z) p_{\theta} (z)$. (5)
> 我们可以将其分解为**似然（likelihood）**和**先验（prior）**：
> $$p_{\theta} (x , z) = p_{\theta} (x |z) p_{\theta} (z). \quad \text{(5)}$$
📌 **分析：**
* **分解联合概率**：这是概率论中的基本规则：联合概率等于条件概率乘以边缘概率。
    * **$p_{\theta} (x |z)$**：给定潜在变量 $z$ 时，数据 $x$ 的**似然**（likelihood），即解码器生成 $x$ 的概率。
    * **$p_{\theta} (z)$**：潜在变量 $z$ 的**先验分布**（prior distribution）。在 VAE 中，我们通常假定 $p_{\theta} (z)$ 服从某个简单的分布（如标准正态分布）。

---

> The goal of the encoder is to map the given observation data x into a multi-dimensional space and obtain its corresponding multi-dimensional feature vector z.
> 编码器的目标是将给定的观测数据 $x$ 映射到多维空间并获得其对应的多维特征向量 $z$。
📌 **分析：**
* 再次强调编码器的作用：从数据中提取特征并进行降维表示。

---

> Thus we can change formula (5) into: $p_{\theta} (z|x) = \frac{p_{\theta} (x|z) p_{\theta}(z)}{p_{\theta} (x)}$. (6)
> 因此，我们可以将公式 (5) 变为：
> $$p_{\theta} (z|x) = \frac{p_{\theta} (x|z) p_{\theta}(z)}{p_{\theta} (x)} \quad \text{(6)}$$
📌 **分析：**
* **贝叶斯定理 (Bayes' Theorem)**：公式 (6) 是贝叶斯定理的直接应用。
    * **$p_{\theta} (z|x)$**：给定数据 $x$ 时，潜在变量 $z$ 的**后验分布**（posterior distribution）。这是我们真正感兴趣的，因为它代表了数据 $x$ 的潜在表示。
    * **$p_{\theta} (x)$**：数据的边缘似然（marginal likelihood），表示观测到数据 $x$ 的总概率。
* 理论上，我们需要计算 $p_{\theta} (z|x)$ 来进行推断，但 $p_{\theta} (x)$ 的计算通常非常困难。

---

> Unfortunately, this integral requires exponential time to compute p(x) as it needs to be evaluated over all configurations of latent variables.
> 不幸的是，由于需要对所有潜在变量配置进行求值，计算 $p(x)$ 的积分需要**指数时间（exponential time）**。
📌 **分析：**
* 指出了计算精确的后验分布 $p_{\theta} (z|x)$（需要 $p_{\theta} (x)$）的困难性：涉及对潜在变量的所有可能配置进行积分，这在实际中是不可行的。这正是 VAE 引入“变分”方法的动机。

---

> We therefore need to approximate this posterior distribution.
> 因此，我们需要**近似（approximate）**这个后验分布。
📌 **分析：**
* 明确了解决计算困难的方法：不直接计算精确的后验分布，而是寻找一个易于处理的近似。

---

> Kingma and Welling [43] introduced a recognition model $q_{\phi}(z|x)$ as an approximation to the intractable true posterior $p_{\theta} (z|x)$, also as a probabilistic encoder.
> Kingma 和 Welling [cite: 43] 引入了一个**识别模型（recognition model）** $q_{\phi}(z|x)$ 作为难以处理的真实后验分布 $p_{\theta} (z|x)$ 的近似，它也是一个**概率编码器（probabilistic encoder）**。
📌 **分析：**
* **“识别模型 $q_{\phi}(z|x)$”**：这就是 VAE 中的“编码器”。它不是直接给出潜在变量 $z$ 的确定性值，而是给出给定数据 $x$ 时 $z$ 的概率分布（例如，均值和方差）。
* **“近似真实后验分布”**：这是 VAE 的核心思想，用一个参数化（由 $\phi$ 参数化）的简单分布 $q_{\phi}(z|x)$ 来近似复杂的真实后验分布 $p_{\theta} (z|x)$。
* **“概率编码器”**：强调了编码器输出的是一个分布，而不是一个单一的向量。

---

> We can use KL divergence to measure the similarity of these two distributionsï¼š $D_{KL} (q_{\phi}(z|x )|| p_{\theta} (z|x )) = \sum_{z∈Z} q_{\phi}(z|x )log \frac{q_{\phi}(z|x )}{p_{\theta} (z |x )} = \sum_{z∈Z} q_{\phi}(z|x )[logq_{\phi}(z|x ) − logp_{\theta} (z|x )] = \sum_{z∈Z} q_{\phi}(z|x )[logq_{\phi}(z|x ) − logp_{\theta} (x , z) + logp_{\theta} (x )] = \sum_{z∈Z} q_{\phi}(z|x )[logq_{\phi}(z|x ) − logp_{\theta} (x , z)] + logp_{\theta} (x ). (7) = E_{q_{\phi}(z|x)}[logq_{\phi}(z|x ) − logp_{\theta} (x , z)] + logp_{\theta} (x )$. (7)
> 我们可以使用 KL 散度来衡量这两个分布的相似性：
> $$D_{KL} (q_{\phi}(z|x )|| p_{\theta} (z|x )) \\ = \sum_{z \in \mathcal{Z}} q_{\phi}(z|x ) \log \frac{q_{\phi}(z|x )}{p_{\theta} (z |x )} \\ = \sum_{z \in \mathcal{Z}} q_{\phi}(z|x )[\log q_{\phi}(z|x ) − \log p_{\theta} (z|x )] \\ = \sum_{z \in \mathcal{Z}} q_{\phi}(z|x )[\log q_{\phi}(z|x ) − \log p_{\theta} (x , z) + \log p_{\theta} (x )] \\ = \sum_{z \in \mathcal{Z}} q_{\phi}(z|x )[\log q_{\phi}(z|x ) − \log p_{\theta} (x , z)] + \log p_{\theta} (x ) \\ = \mathbb{E}_{q_{\phi}(z|x)}[\log q_{\phi}(z|x ) − \log p_{\theta} (x , z)] + \log p_{\theta} (x ). \quad \text{(7)}$$
📌 **分析：**
* **KL散度作为优化目标**：为了让 $q_{\phi}(z|x)$ 尽可能接近 $p_{\theta} (z|x)$，我们最小化两者之间的 KL 散度。公式 (7) 展示了 KL 散度的展开过程，最终将其与 $p_{\theta} (x)$（数据的边缘似然）以及一个期望项联系起来。
* **变分推断**：这是 VAE 的核心数学推导，将难以计算的后验分布问题转化为可优化的 KL 散度问题。

---

> Then we can get: $logp_{\theta} (x ) = E_{q_{\phi}(z|x)}[logp_{\theta} (x , z) − logq_{\phi}(z|x )] + D_{KL}( p_{\theta} (z|x )||q_{\phi}(z|x )). (8)$
> 那么我们可以得到：
> $$\log p_{\theta} (x ) = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta} (x , z) − \log q_{\phi}(z|x )] + D_{KL}( p_{\theta} (z|x )||q_{\phi}(z|x )). \quad \text{(8)}$$
📌 **分析：**
* 公式 (8) 是从公式 (7) 重新排列得到的。它表示了数据边缘似然 $\log p_{\theta} (x)$ 可以被分解为两部分：
    * 第一部分是**变分下界（Evidence Lower Bound, ELBO）**，通常称为 $L(\theta, \phi)$。
    * 第二部分是真实后验分布 $p_{\theta} (z|x)$ 与近似后验分布 $q_{\phi}(z|x)$ 之间的 KL 散度。

---

> The second term is the KL divergence of the approximate from the true posterior. Since this KL divergence is non-negative, then: $logp_{\theta} (x ) ≥ E_{q_{\phi}(z|x)}[−logq_{\phi}(z|x ) + logp_{\theta} (x , z)] = E_{q_{\phi}(z|x)}[−logq_{\phi}(z|x ) + logp_{\theta} (x |z) + logp_{\theta} (z)] = E_{q_{\phi}(z|x)}[logp_{\theta} (x |z)] − D_{KL} (q_{\phi}(z|x )|| p_{\theta} (z)) = L(θ, φ). (9)$
> 第二项是近似分布与真实后验分布的KL散度。由于这个KL散度是非负的，那么：
> $$\log p_{\theta} (x ) \ge \mathbb{E}_{q_{\phi}(z|x)}[-\log q_{\phi}(z|x ) + \log p_{\theta} (x , z)] \\ = \mathbb{E}_{q_{\phi}(z|x)}[-\log q_{\phi}(z|x ) + \log p_{\theta} (x |z) + \log p_{\theta} (z)] \\ = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta} (x |z)] − D_{KL} (q_{\phi}(z|x )|| p_{\theta} (z)) \\ = L(\theta, \phi). \quad \text{(9)}$$
📌 **分析：**
* **非负性**：KL 散度总是大于等于 0。
* **变分下界 (ELBO)**：由于 $D_{KL}(p_{\theta}(z|x)||q_{\phi}(z|x))$ 是非负的，因此公式 (8) 中的第一项 $L(\theta, \phi)$ 就成为了 $\log p_{\theta} (x)$ 的下界。
* **VAE 的目标**：我们无法直接最大化 $\log p_{\theta} (x)$，但可以通过最大化其下界 $L(\theta, \phi)$ 来间接优化模型。
* **$L(\theta, \phi)$ 的两部分**：
    * **$\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta} (x |z)]$**：这是**重构项（reconstruction term）**，它鼓励解码器在给定潜在表示 $z$ 的情况下，能够很好地重构原始数据 $x$。这一项对应于文本的**感知不可察觉性（perceptual-imperceptibility）**。
    * **$-D_{KL} (q_{\phi}(z|x )|| p_{\theta} (z))$**：这是**正则化项（regularization term）**，它鼓励编码器学习到的潜在分布 $q_{\phi}(z|x)$ 尽可能接近预设的先验分布 $p_{\theta} (z)$（通常是标准正态分布）。这对应于文本的**统计不可察觉性（statistical-imperceptibility）**。
* 这个损失函数的设计正是 VAE-Stega 能够同时平衡和优化两种不可察觉性的关键。

---

> Here, $L(\theta, \phi)$ is called the lower bound on the marginal likelihood of datapoint x.
> 这里，$L(\theta, \phi)$ 被称为数据点 $x$ 的**边缘似然（marginal likelihood）**的下界。
📌 **分析：**
* 进一步明确了 $L(\theta, \phi)$ 的术语和意义。

---

> If we want to maximize the log-likelihood function $logp_{\theta}(x).$, we can maximize the variation lower bound $L(\theta, \phi)$.
> 如果我们想最大化对数似然函数 $log p_{\theta}(x)$，我们可以最大化变分下界 $L(\theta, \phi)$。
📌 **分析：**
* 总结了 VAE 的训练策略：通过最大化 ELBO 来间接最大化数据的真实似然。

---

> The loss function of the variational auto-encoder is the negative log-likelihood with a regularizer.
> 变分自编码器的损失函数是带有正则化项的负对数似然。
📌 **分析：**
* 解释了 VAE 的损失函数形式：通常是将 ELBO 的负值作为损失函数，然后最小化它。

---

> The total loss is then $\sum_{i=1}^{N}Loss_i$ for N total datapoints.
> 则对于 $N$ 个数据点，总损失为 $\sum_{i=1}^{N}Loss_i$。
📌 **分析：**
* 解释了 VAE 的训练过程，对所有数据点计算损失并求和。

---

> The loss function for datapoint x is: $Loss = −L(θ, φ) = −\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta} (x |z)] + D_{KL} (q_{\phi}(z|x )|| p_{\theta}(z))$. (10)
> 数据点 $x$ 的损失函数为：
> $$Loss = −L(\theta, \phi) = −\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta} (x |z)] + D_{KL} (q_{\phi}(z|x )|| p_{\theta}(z)). \quad \text{(10)}$$
📌 **分析：**
* **VAE 损失函数（最终形式）**：这是 VAE 训练的核心目标函数。
    * **第一项：$- \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta} (x |z)]$**
        * 这是**重构损失（reconstruction loss）**的负值（或负对数似然，等价于最小化负对数似然，即最大化似然）。它衡量了解码器从潜在表示 $z$ 重构出原始数据 $x$ 的能力。这一项对应于提高生成文本的**感知不可察觉性（perceptual-imperceptibility）**。
    * **第二项：$+ D_{KL} (q_{\phi}(z|x )|| p_{\theta}(z))$**
        * 这是**正则化损失（regularizer loss）**，即编码器输出的潜在分布 $q_{\phi}(z|x)$ 与预设的先验分布 $p_{\theta}(z)$ 之间的 KL 散度。这一项旨在确保编码器学习到的潜在空间是连续的、结构化的，并且其分布与预设的先验分布相似。这直接对应于提高生成文本的**统计不可察觉性（statistical-imperceptibility）**。
* 这个损失函数的设计正是 VAE-Stega 能够同时平衡和优化两种不可察觉性的关键。

---

> In equation (10), $q_{\phi}(z|x)$ can be regarded as an encoder, which maps the training samples to the latent spaces Z.
> 在公式 (10) 中，$q_{\phi}(z|x)$ 可以被视为一个编码器，它将训练样本映射到潜在空间 $Z$。
📌 **分析：**
* 再次明确了 $q_{\phi}(z|x)$ 的角色：概率编码器。

---

> $p_{\theta} (x|z)$ can be regarded as a decoder, which samples the latent vector z from the latent space Z and then generates new samples.
> $p_{\theta} (x|z)$ 可以被视为一个解码器，它从潜在空间 $Z$ 中采样潜在向量 $z$，然后生成新的样本。
📌 **分析：**
* 再次明确了 $p_{\theta} (x|z)$ 的角色：概率解码器。

---

> The first term of equation (10) is the reconstruction loss, which encourages the decoder to learn to generate new sentences from latent space Z.
> 公式 (10) 的第一项是**重构损失（reconstruction loss）**，它鼓励解码器学习从潜在空间 $Z$ 生成新句子。
📌 **分析：**
* 重构损失的目标是使生成的 $x'$ 尽可能接近原始 $x$，确保生成文本的质量和自然度。

---

> The second term is the regularizer loss, which is the Kullback-Leibler divergence between the encoder’s distribution $q_{\phi}(z|x)$ and $p_{\theta} (z)$.
> 第二项是**正则化损失（regularizer loss）**，它是编码器分布 $q_{\phi}(z|x)$ 和 $p_{\theta} (z)$ 之间的 Kullback-Leibler 散度。
📌 **分析：**
* 正则化损失确保了潜在空间的分布符合先验假设，从而使生成的文本具有与正常文本相似的整体统计特性。

---

> It is worth noting that $p_{\theta} (z)$ acts as a regularizer on the overall distribution learned by the encoder, so in fact it can be any distribution function.
> 值得注意的是，$p_{\theta} (z)$ 作为对编码器学习到的整体分布的**正则化项（regularizer）**，因此实际上它可以是任何分布函数。
📌 **分析：**
* **“正则化项”**：用于防止过拟合，并引导模型学习到期望的特性。
* **“可以是任何分布函数”**：理论上如此，但为了数学上的便利和收敛性，通常会选择简单的分布。

---

> Here, in order to simplify the calculation of KL divergence, we follow Kingma and Welling [43] and set it to be the standard Normal distribution with mean zero and variance one, that is $Normal(0, 1)$.
> 在这里，为了简化 KL 散度的计算，我们遵循 Kingma 和 Welling [cite: 43] 的做法，将其设置为均值为零、方差为一的**标准正态分布（standard Normal distribution）**，即 $Normal(0, 1)$。
📌 **分析：**
* **标准正态分布 $Normal(0,1)$**：这是 VAE 中最常用的先验分布假设。选择它的原因主要是其数学性质良好，KL 散度计算有解析解，便于优化。这意味着编码器被强制将所有输入样本映射到潜在空间中的一个符合标准正态分布的区域。

---

> B. Encoder in VAE-Stega
> B. **VAE-Stega 中的编码器**
📌 **分析：**
* 本小节将详细介绍 VAE-Stega 模型中编码器的具体实现。

---

> In VAE architecture, any model can be used as an encoder as long as it can extract the feature expression of the input sentence and map it to the feature space Z.
> 在 VAE 架构中，只要模型能够提取输入语句的特征表达并将其映射到特征空间 $Z$，任何模型都可以用作编码器。
📌 **分析：**
* 强调了 VAE 架构的灵活性和模块化：编码器可以独立于解码器进行设计和选择，只要它能完成特征提取和映射的任务。

---

> In this paper, we designed and compared two different encoders, one of them use a recurrent neural network with LSTM units as the encoder, we call it VAE-Stega (LSTM-LSTM).
> 在本文中，我们设计并比较了两种不同的编码器，其中一种使用带有 **LSTM 单元（LSTM units）**的**循环神经网络（recurrent neural network）**作为编码器，我们称之为 **VAE-Stega (LSTM-LSTM)**。
📌 **分析：**
* **“LSTM 单元”**：长短期记忆网络 (Long Short-Term Memory) 是一种特殊的 RNN 单元，能够有效解决传统 RNN 中的梯度消失和梯度爆炸问题，从而更好地处理长序列的依赖关系。
* **VAE-Stega (LSTM-LSTM)**：表示模型中编码器和解码器都使用 LSTM。这是一种相对传统的序列建模方法。

---

> The other use Bidirectional Encoder Representations from Transformers (BERT) [48] as the encoder, we call it VAE-Stega (BERT-LSTM).
> 另一种使用基于 **Transformer** 的**双向编码器表示（Bidirectional Encoder Representations from Transformers, BERT）** [cite: 48] 作为编码器，我们称之为 **VAE-Stega (BERT-LSTM)**。
📌 **分析：**
* **“BERT”** [cite: 48]：由 Google 提出的一种强大的预训练语言模型，它基于 Transformer 架构，并通过双向训练（即同时考虑一个词的左右上下文）来学习深层语境表示。BERT 在多种 NLP 任务中都取得了SOTA（State-of-the-Art）性能。
* **VAE-Stega (BERT-LSTM)**：表示模型中使用 BERT 作为编码器，LSTM 作为解码器。这种组合旨在利用 BERT 强大的特征提取能力。

---

> Next, we will first introduce VAE-Stega (BERT-LSTM) in detail, whose structure has been shown in Figure 5.
> 接下来，我们将首先详细介绍 VAE-Stega (BERT-LSTM)，其结构已在图5中显示。
📌 **分析：**
* 引导读者查看图5，并说明将重点介绍 BERT 作为编码器的版本。
![[Pasted image 20250521150744.png]]
---

> BERT is a pre-trained text feature extraction model based on transformer structure [49], due to its remarkable ability of text feature extraction and expression, BERT has been widely used in the field of natural language processing.
> BERT 是一种基于 **Transformer 结构** [cite: 49] 的**预训练文本特征提取模型（pre-trained text feature extraction model）**，由于其卓越的文本特征提取和表达能力，BERT 在自然语言处理领域得到了广泛应用。
📌 **分析：**
* **“Transformer 结构”** [cite: 49]：一种基于注意力机制（attention mechanism）的神经网络架构，彻底改变了序列建模领域。它克服了 RNN 难以并行化和处理长距离依赖的缺点。
* **“预训练文本特征提取模型”**：BERT 在大规模语料库上进行了预训练，学习了丰富的语言知识和上下文信息，因此能够提取高质量的文本特征。
* 再次强调了 BERT 的强大能力，解释了选择它作为编码器的理由。

---

> The transformer architechture is composed of a stack of identical layers, each of which contains multi-head self-attention sublayer and a position-wise fully-connected sublayer.
> Transformer 架构由堆叠的相同层组成，每层包含一个**多头自注意力（multi-head self-attention）子层**和一个**逐位置全连接（position-wise fully-connected）子层**。
📌 **分析：**
* 简要介绍了 Transformer 的核心组成部分：
    * **多头自注意力子层**：Transformer 最关键的创新。它允许模型同时关注输入序列中不同位置的信息，并为不同“方面”的信息分配不同的注意力权重。
    * **逐位置全连接子层**：一个简单的全连接前馈网络，独立应用于序列中的每个位置。

---

> For each independent sentence x in a big training corpus, it can be regarded as a sequential signal and the i -th word in x can be viewed as the signal at the time point i , that is $x = \{x_1, x_2, x_3, . . . , x_n\}, ∀ x_j ∈ D$ (11)
> 对于大型训练语料库中的每个独立句子 $x$，它可以被视为一个序列信号，其中 $x$ 中的第 $i$ 个词可以被视为时间点 $i$ 的信号，即：
> $$x = \{x_1, x_2, x_3, . . . , x_n\}, \forall x_j \in D \quad \text{(11)}$$
📌 **分析：**
* 再次强调了句子作为序列信号的建模方式。

---

> where $x_j$ indicates the i -th word in sentence x and n is the length of it, D is the dictionary which contains all possible words.
> 其中 $x_j$ 表示句子 $x$ 中的第 $i$ 个词，$n$ 是其长度，$D$ 是包含所有可能词的字典。
📌 **分析：**
* 对公式 (11) 中符号的解释，再次强调了词典 $D$ 和句子长度 $n$ 的概念。

---

> We map each word in x to a dense semantic space with a dimension of d, that is $x_i ∈ R^d$.
> 我们将 $x$ 中的每个词映射到维度为 $d$ 的**密集语义空间（dense semantic space）**，即 $x_i ∈ \mathbb{R}^d$。
📌 **分析：**
* **“密集语义空间”**：指词向量（word embedding）空间。每个词被表示为一个稠密的实数向量，这些向量捕捉了词的语义信息，相似的词在向量空间中距离更近。这是现代 NLP 的基础。

---

> To consider the order of words in the input sentence, similar to the method proposed in [49], we use a position encoding module to inject the information about the position of each word into the proposed model, that is: { $P E_{(i,2 j)} = si n(i /10000^{2 j/d})$, $P E_{(i,2 j+1)} = cos(i /10000^{2 j/d})$, (12)
> 为了考虑输入句子中词的顺序，类似于 [cite: 49] 中提出的方法，我们使用一个**位置编码（position encoding）模块**将每个词的位置信息注入到所提出的模型中，即：
> $$\begin{cases} PE_{(i,2j)} = \sin(i /10000^{2j/d}), \\ PE_{(i,2j+1)} = \cos(i /10000^{2j/d}), \end{cases} \quad \text{(12)}$$
📌 **分析：**
* **“位置编码”** [cite: 49]：Transformer 架构的一项关键技术。由于 Transformer 内部的自注意力机制是并行处理的，不包含序列信息，因此需要显式地将词的位置信息编码并加到词嵌入中，以保留词序信息。公式 (12) 给出了正弦和余弦函数形式的位置编码。
    * $i$: 词在句子中的索引。
    * $j$: 词嵌入向量中的维度索引。

---

> where i is the word index in input sentence, j is the j -th value in the embedding vector of $x_i$. The position encodings are simply added to the input embeddings to encode position information.
> 其中 $i$ 是输入语句中的词索引，$j$ 是 $x_i$ 的嵌入向量中的第 $j$ 个值。位置编码简单地添加到输入嵌入中以编码位置信息。
📌 **分析：**
* 进一步解释了位置编码的机制：直接与词嵌入相加，从而使模型能够区分相同词在不同位置时的含义。

---

> In order to get a more accurate semantic representation of $x_i$, we need to consider the language environment of it.
> 为了获得更准确的 $x_i$ 的语义表示，我们需要考虑它的语言环境。
📌 **分析：**
* 引入了上下文信息的重要性，这也是自注意力机制的核心。

---

> Therefore we try to learn the semantic correlations between $x_i$ and other words in the input sentence $x$.
> 因此，我们尝试学习输入句子 $x$ 中 $x_i$ 与其他词之间的语义关联。
📌 **分析：**
* 解释了为什么需要自注意力机制：为了捕捉词语之间的依赖关系，从而获得上下文感知的语义表示。

---

> For example, the correlation between $x_i$ and $x_j$ under a specific attention head k can be calculated as follows: $α^{(k)}_{i, j} = \frac{exp(φ^{(k)}(x_i , x_j ))}{\sum_{j=1}^n exp(φ^{(k)}(x_i , x_j ))}$, $φ^{(k)}(x_i , x_j ) = \frac{1}{\sqrt{d_s}} (W_{query}^{(k)}x_i )^T (W_{key}^{(k)}x_j )$, (13)
> 例如，在特定注意力头 $k$ 下，$x_i$ 和 $x_j$ 之间的关联可以按如下方式计算：
> $$α^{(k)}_{i, j} = \frac{\exp(φ^{(k)}(x_i , x_j ))}{\sum_{j=1}^n \exp(φ^{(k)}(x_i , x_j ))} \\ φ^{(k)}(x_i , x_j ) = \frac{1}{\sqrt{d_s}} (W_{query}^{(k)}x_i )^T (W_{key}^{(k)}x_j ) \quad \text{(13)}$$
📌 **分析：**
* **自注意力机制（Self-Attention）**的核心计算：
    * **$φ^{(k)}(x_i , x_j )$**：计算查询向量 (Query) 和键向量 (Key) 的点积，衡量两个词 $x_i$ 和 $x_j$ 之间的“相关性分数”。这里的 $W_{query}^{(k)}$ 和 $W_{key}^{(k)}$ 是可学习的权重矩阵，用于将词嵌入转换到不同的表示空间。
    * **$\frac{1}{\sqrt{d_s}}$**：缩放因子，用于防止点积过大，导致 softmax 梯度过小。
    * **$α^{(k)}_{i, j}$**：通过 softmax 函数归一化后的注意力权重，表示词 $x_j$ 对词 $x_i$ 的重要性。$\sum_{j=1}^n \exp(φ^{(k)}(x_i , x_j ))$ 是归一化项。
* 公式 (13) 描述了 Transformer 中单个注意力头的计算过程。

---

> where $d_s = d_h/K$, $d_h$ is the dimension of hidden states and $K$ is the number of attention heads. $W_{query}^{(k)}$, $W_{key}^{(k)} ∈ R^{d_s×d_h}$ are transformation matrixes which map the words in sentence $x$ into the feature space $R^{d_h}$.
> 其中 $d_s = d_h/K$， $d_h$ 是隐藏状态的维度，$K$ 是注意力头的数量。$W_{query}^{(k)}$, $W_{key}^{(k)} \in \mathbb{R}^{d_s \times d_h}$ 是将句子 $x$ 中的词映射到特征空间 $\mathbb{R}^{d_h}$ 的转换矩阵。
📌 **分析：**
* 解释了公式 (13) 中符号的含义：
    * **$d_s$**：单个注意力头的维度。
    * **$d_h$**：隐藏层维度。
    * **$K$**：多头注意力的头数。
    * **$W_{query}^{(k)}$, $W_{key}^{(k)}$**：可学习的权重矩阵，将输入词嵌入转换为查询 (Query) 和键 (Key) 向量。

---

> Next, we update the representation of feature $x_i$ in subspace k via combining all relevant features guided by coefficients $α^{(k)}_{i, j}$ : $s^{(k)}_i = \sum_{j=1}^n α^{(k)}_{i, j} (W_{value}^{(k)}x_j )$, (14)
> 接下来，我们通过结合由系数 $α^{(k)}_{i, j}$ 指导的所有相关特征，更新子空间 $k$ 中特征 $x_i$ 的表示：
> $$s^{(k)}_i = \sum_{j=1}^n α^{(k)}_{i, j} (W_{value}^{(k)}x_j ) \quad \text{(14)}$$
📌 **分析：**
* **“更新特征表示”**：根据计算出的注意力权重 $α^{(k)}_{i, j}$，将所有词的**值向量（Value）**的加权和作为当前词 $x_i$ 在该注意力头下的新表示 $s^{(k)}_i$。
* **$W_{value}^{(k)}$**：可学习的权重矩阵，用于将词嵌入转换为值 (Value) 向量。
* 公式 (14) 展示了单个注意力头如何计算输出。

---

> where $W_{value}^{(k)} ∈ R^{d_s×d_h}$ denotes the correlation matrix.
> 其中 $W_{value}^{(k)} ∈ \mathbb{R}^{d_s \times d_h}$ 表示相关矩阵。
📌 **分析：**
* 解释了公式 (14) 中 $W_{value}^{(k)}$ 的含义。

---

> The above operations are just under one attention head. To help the model jointly attend to information from different representation subspaces at different positions, we conduct **multi-head attention** in the proposed model.
> 上述操作仅在一个注意力头下进行。为了帮助模型共同关注来自不同表示子空间、不同位置的信息，我们在所提出的模型中进行**多头注意力（multi-head attention）**。
📌 **分析：**
* **“多头注意力”**：Transformer 的另一个关键特性。它并行运行多个独立的注意力头，每个头学习不同的“方面”或“关系”。最后将所有头的输出拼接起来，从而使模型能够从多个角度捕捉上下文信息，丰富其表示能力。

---

> After that, we concatenate the output of each attention head, and then take the weighted output as the whole output, that is: $s_i = W_{out} [s^{(1)}_i ⊕ s^{(2)}_i ⊕ . . . ⊕ s^{(K)}_i ]$. (15)
> 之后，我们将每个注意力头的输出**拼接（concatenate）**起来，然后将加权输出作为整个输出，即：
> $$s_i = W_{out} [s^{(1)}_i ⊕ s^{(2)}_i ⊕ . . . ⊕ s^{(K)}_i ]. \quad \text{(15)}$$
📌 **分析：**
* **“拼接（$\oplus$）”**：将不同注意力头的输出向量在维度上连接起来。
* **$W_{out}$**：一个可学习的权重矩阵，用于将拼接后的多头注意力输出线性变换回预期的维度。
* 公式 (15) 描述了多头注意力如何整合来自不同头的输出。

---

> Here, $⊕$ is the concatenation operator, $W_{out}$ denotes the weight matrix.
> 这里，$⊕$ 是拼接运算符，$W_{out}$ 表示权重矩阵。
📌 **分析：**
* 对公式 (15) 中符号的解释。

---

> Then the output of the multi-head self-atttention sublayer will be $S = \{s_1, s_2, . . . , s_n\}$. We employ a residual connection [50] which denoted as R() and layer normalization [51] on the output of each layer, then the processed output can be denoted as follows: $S ̃ = \{s ̃_1, s ̃_2, . . . , s ̃_n\} = LayerNor m(R(S) + S)$. (16)
> 那么多头自注意力子层的输出将是 $S = \{s_1, s_2, . . . , s_n\}$。我们对每层的输出采用一个**残差连接（residual connection）** [cite: 50]（表示为 $R()$）和**层归一化（layer normalization）** [cite: 51]，然后处理后的输出可以表示为：
> $$\tilde{S} = \{\tilde{s}_1, \tilde{s}_2, . . . , \tilde{s}_n\} = \text{LayerNorm}(R(S) + S). \quad \text{(16)}$$
📌 **分析：**
* **“残差连接”** [cite: 50]：一种跳跃连接，将层的输入直接加到层的输出上。这有助于缓解深度神经网络中的梯度消失问题，使模型能够训练得更深。
* **“层归一化”** [cite: 51]：一种归一化技术，对每个样本的特征进行归一化。这有助于稳定训练过程，加速收敛，并提高模型的泛化能力。
* 公式 (16) 展示了 Transformer 块中常见的残差连接和层归一化操作，它们是深度神经网络训练的关键技术。

---

> We send this to the fully-connected sublayer, and the outputs are as follows: { $O = \{o_1, o_2, . . . , o_n\}$, $o_i = W_2 ReLU (W_1\tilde{s}_i + b_1) + b_2$, (17)
> 我们将其送入全连接子层，输出如下：
> $$\begin{cases} O = \{o_1, o_2, . . . , o_n\}, \\ o_i = W_2 \text{ReLU} (W_1\tilde{s}_i + b_1) + b_2, \end{cases} \quad \text{(17)}$$
📌 **分析：**
* **“全连接子层”**：Transformer 块中的另一个核心组件，它对序列中的每个位置独立地应用一个两层的前馈网络。
* **ReLU**：非线性激活函数，引入非线性能力。
* 公式 (17) 描述了 Transformer 块中的全连接层计算。

---

> where $W_1, W_2$ and $b_1, b_2$ are learned weight matrix and bias. Similarly, residual connection and layer normalization are applied to the output. The finnal output of this transformer structure is denoted as follows: $O ̃ = \{o ̃_1, o ̃_2, . . . , o ̃_n\} = LayerNor m(S ̃ + O)$. (18)
> 其中 $W_1, W_2$ 和 $b_1, b_2$ 是学习到的权重矩阵和偏置。类似地，残差连接和层归一化也应用于输出。该 Transformer 结构的最终输出表示如下：
> $$\tilde{O} = \{\tilde{o}_1, \tilde{o}_2, . . . , \tilde{o}_n\} = \text{LayerNorm}(\tilde{S} + O). \quad \text{(18)}$$
📌 **分析：**
* 再次应用了残差连接和层归一化，这是 Transformer 每一子层后的标准操作。
* 公式 (18) 表示一个完整的 Transformer 块的最终输出。

---

> BERT is based on deep bidirectional transformers, each of which will complete the semantic extraction operation of the input sentence as described above.
> BERT 基于**深度双向 Transformer（deep bidirectional transformers）**，每个 Transformer 都将完成上述输入句子的语义提取操作。
📌 **分析：**
* **“深度双向 Transformer”**：强调了 BERT 的两个关键特性：
    * **深度**：由多层 Transformer 堆叠而成。
    * **双向**：在预训练时，BERT 能够同时考虑一个词的左侧和右侧上下文，从而学习到更全面的语境表示。
* 总结了 BERT 作为编码器的作用：提取输入句子的丰富语义特征。

---

> Finally, we use the output of last layer in BERT as the contextual feature representation of each input sentence, that is: $z(x) = B E RT (x)$. (19)
> 最后，我们使用 BERT 中最后一层的输出作为每个输入语句的**上下文特征表示（contextual feature representation）**，即：
> $$z(x) = \text{BERT}(x). \quad \text{(19)}$$
📌 **分析：**
* **“上下文特征表示”**：BERT 的核心优势之一。它为句子中的每个词生成一个向量，这个向量不仅包含词本身的含义，还包含了它在特定句子语境中的含义。
* 公式 (19) 简洁地表示了 BERT 作为编码器的输出：将输入句子 $x$ 映射为它的上下文特征表示 $z(x)$。这个 $z(x)$ 将用于后续计算潜在空间分布的均值和方差。

---

> Since we set $p_{\theta} (z)$ to be the standard Normal distribution $Normal(0, 1)$, then, it requires the distribution of training samples in the hidden space ($q_{\phi}(z|x)$ in formula (10)) also obeys the normal distribution as much as possible.
> 由于我们将 $p_{\theta} (z)$ 设置为标准正态分布 $Normal(0, 1)$，那么，这要求训练样本在隐藏空间（公式 (10) 中的 $q_{\phi}(z|x)$）中的分布也尽可能服从正态分布。
📌 **分析：**
* **先验分布约束**：再次强调了 VAE 的设计原则：强制编码器输出的潜在分布 $q_{\phi}(z|x)$ 尽可能接近预设的先验分布 $p_{\theta} (z)$，这里是标准正态分布。
* 这样做的好处是使得潜在空间具有良好的连续性和光滑性，方便采样和生成。

---

> We assume that $q_{\phi}(z|x)$ follows the normal distribution with the mean vector $\vec{\mu}$ and the standard deviation vector $\vec{\sigma}$, that is, $q_{\phi}(z|x) = N(\vec{\mu} , \vec{\sigma})$.
> 我们假设 $q_{\phi}(z|x)$ 遵循均值向量 $\vec{\mu}$ 和标准差向量 $\vec{\sigma}$ 的正态分布，即 $q_{\phi}(z|x) = N(\vec{\mu} , \vec{\sigma})$。
📌 **分析：**
* **参数化编码器**：明确了编码器输出的潜在分布是**高斯分布**（正态分布），这意味着编码器需要学习每个输入样本对应的均值向量 $\vec{\mu}$ 和标准差向量 $\vec{\sigma}$。

---

> We use Highway network [52] to learn vector $\vec{\mu}$ and vector $\vec{\sigma}$: { $\vec{\mu} = H (W_{\mu}, z(x)) \cdot T (W_{\mu}, z(x)) + z(x)(1−T (W_{\mu}, z(x)))$, $\vec{\sigma} = H (W_{\sigma} , z(x)) \cdot T (W_{\sigma} , z(x)) + z(x)(1−T(W_{\sigma} , z(x)))$. (20)
> 我们使用 **Highway network** [cite: 52] 来学习向量 $\vec{\mu}$ 和向量 $\vec{\sigma}$：
> $$\begin{cases} \vec{\mu} = H (W_{\mu}, z(x)) \cdot T (W_{\mu}, z(x)) + z(x)(1−T (W_{\mu}, z(x))), \\ \vec{\sigma} = H (W_{\sigma} , z(x)) \cdot T (W_{\sigma} , z(x)) + z(x)(1−T(W_{\sigma} , z(x))). \end{cases} \quad \text{(20)}$$
📌 **分析：**
* **“Highway network”** [cite: 52]：一种具有门控机制的神经网络结构，允许信息在层之间“直接传递”（通过门）。它借鉴了 LSTM 的门控思想，可以更好地控制信息流，有助于训练更深的网络。
* **学习 $\vec{\mu}$ 和 $\vec{\sigma}$**：BERT 的输出 $z(x)$ 经过 Highway network 转换，生成 $\vec{\mu}$ 和 $\vec{\sigma}$。
* **$H()$ 和 $T()$**：
    * $H()$ 是一个非线性变换函数（如 ReLU 激活的全连接层）。
    * $T()$ 是一个**变换门（transform gate）**，通常是一个 sigmoid 激活的全连接层，其输出范围在 0 到 1 之间，用于控制有多少信息通过 $H()$ 转换，有多少信息直接从输入 $z(x)$ 传递。
* 公式 (20) 描述了如何从 BERT 的上下文特征表示 $z(x)$ 得到潜在分布的均值和方差。

---

> where $W_{\mu}$ and $W_{\sigma}$ are the learnt weights. $T ()$ is the transform gate which can be define as $T (A) = f (W_T A + b_T )$, where $W_T$ is the weight matrix and $b_T$ is the bias vector, $f ()$ is a nonlinear activation function.
> 其中 $W_{\mu}$ 和 $W_{\sigma}$ 是学习到的权重。$T ()$ 是变换门，可以定义为 $T (A) = f (W_T A + b_T )$，其中 $W_T$ 是权重矩阵，$b_T$ 是偏置向量，$f ()$ 是非线性激活函数。
📌 **分析：**
* 进一步解释了 Highway network 中权重和变换门的定义。

---

> In order to ensure the statistical-imperceptibility of the generated steganographic sentences, we hope that the new generated steganographic sentences also conform to such overall distribution characteristic in the feature space.
> 为了确保生成的隐写语句的统计不可察觉性，我们希望新生成的隐写语句在特征空间中也符合这种整体分布特征。
📌 **分析：**
* 重申了使用 VAE 编码器的目的：确保生成的文本在潜在特征空间中的分布与正常文本的分布一致，从而达到统计不可察觉性。

---

> To achieve this, we consider to first randomly sample latent vector z from the space according to $p_Z$.
> 为了实现这一点，我们考虑首先根据 $p_Z$ 从空间中随机采样潜在向量 $z$。
📌 **分析：**
* **生成过程**：明确了在生成隐写文本时，是从潜在空间中随机采样 $z$。这个 $p_Z$ 实际上就是我们假设的先验分布 $p_{\theta}(z)$（即 $Normal(0,1)$）。

---

> Then we use the decoder to generate sentences under the constraint of z, thus can keep the feature distribution of generated sentences conforms to that of normal sentences.
> 然后我们使用解码器在 $z$ 的约束下生成句子，从而使生成句子的特征分布符合正常句子的特征分布。
📌 **分析：**
* 总结了编码器部分如何与解码器连接：采样到的潜在向量 $z$ 将作为解码器生成文本的起始条件或约束，以确保生成文本的整体统计特性与正常文本的潜在分布一致。

> C. Decoder in VAE-Stega
> C. **VAE-Stega 中的解码器**
📌 **分析：**
* 本小节将详细介绍 VAE-Stega 模型中解码器的具体实现，以及它如何与编码器和信息隐藏机制协同工作。

---

> For a sentence x with length n, which can be represented as $x = \{x_1, x_2, . . . , x_n\}$, $x_i ∈ D$. The task of decoder is to find a suitable word sequence with complete semantics and correct syntax among $N^n$ possible combinations according to the latent vector z sampled from the latent space Z.
> 对于一个长度为 $n$ 的句子 $x$，它可以表示为 $x = \{x_1, x_2, . . . , x_n\}$，其中 $x_i \in D$。解码器的任务是根据从潜在空间 $Z$ 中采样的潜在向量 $z$，在 $N^n$ 种可能的组合中找到一个具有完整语义和正确语法的合适词序列。
📌 **分析：**
* 明确了解码器的核心任务：从一个巨大的可能性空间中，根据潜在向量 $z$ 的指导，生成一个既符合语义又符合语法的自然语言句子。这体现了生成式模型“创造”新内容的能力。

---

> Referring to the model proposed in [13], [47], in this work, we use Recurrent Neural Network (RNN) as decoder for steganographic sentences generation.
> 参考 [13]、[47] 中提出的模型，在这项工作中，我们使用**循环神经网络（Recurrent Neural Network, RNN）**作为隐写语句生成的解码器。
📌 **分析：**
* **“循环神经网络（RNN）”**：一种适合处理序列数据的神经网络，通过在时间步之间共享参数和传递隐藏状态来捕捉序列中的依赖关系。RNN，特别是其变体（如LSTM、GRU），在语言建模和文本生成中表现出色。
* 选择 RNN 作为解码器是常见的做法，因为它能够学习序列的上下文依赖性。

---

> We use LSTM unit as the nonlinear transformation function of the hidden layer in recurrent neural network, which can be described using the following formulas: { $I_t = σ (W_i · [h_{t−1}, x_t ] + b_i )$, $F_t = σ (W_f · [h_{t−1}, x_t ] + b_f )$, $C_t = F_t · C_{t−1} + I_t · tanh(W_c · [h_{t−1}, x_t ] + b_c)$, $O_t = σ (W_o · [h_{t−1}, x_t ] + b_o)$, $h_t = O_t · tanh(C_t )$. (21)
> 我们使用 **LSTM 单元（LSTM unit）**作为循环神经网络隐藏层的非线性变换函数，这可以用以下公式描述：
> $$\begin{cases} I_t = \sigma (W_i \cdot [h_{t−1}, x_t ] + b_i ), \\ F_t = \sigma (W_f \cdot [h_{t−1}, x_t ] + b_f ), \\ C_t = F_t \cdot C_{t−1} + I_t \cdot \tanh(W_c \cdot [h_{t−1}, x_t ] + b_c), \\ O_t = \sigma (W_o \cdot [h_{t−1}, x_t ] + b_o), \\ h_t = O_t \cdot \tanh(C_t ). \end{cases} \quad \text{(21)}$$
📌 **分析：**
* **“LSTM 单元”**：长短期记忆网络 (Long Short-Term Memory) 是一种特殊的 RNN 单元，旨在解决传统 RNN 在处理长序列时可能出现的梯度消失或梯度爆炸问题，从而更好地捕捉长距离依赖。
* **门控机制**：公式 (21) 详细展示了 LSTM 的内部结构，包括：
    * **输入门 ($I_t$)**：控制有多少新的输入信息可以更新记忆单元。
    * **遗忘门 ($F_t$)**：控制有多少过去的记忆信息应该被遗忘。
    * **记忆单元 ($C_t$)**：存储长期信息，它通过遗忘门和输入门的协同作用来更新。
    * **输出门 ($O_t$)**：控制有多少记忆单元的信息可以输出到隐藏状态。
    * **隐藏状态 ($h_t$)**：当前时间步的输出。
* **$\sigma$ (sigmoid 函数)**：将值压缩到0到1之间，作为门的激活函数。
* **tanh (双曲正切函数)**：将值压缩到-1到1之间，用于转换输入。
* 通过这些门，LSTM 能够选择性地记住或遗忘信息，使其在处理文本序列时具有更强的记忆能力。

---

> where $I_t , F_t , O_t$ indicate the input gate, the forget gate and the output gate, separately. $W_i , W_f , W_c, W_o$ are the weights in them and $b_i , b_f , b_c, b_o$ are the bias. $h_t$ stands for the hidden state at time step t. The memory cell $C_t$ is a summation of the incoming information modulated by the input gate and previous memory modulated by the forget gate $F_t$. For simplicity, we denote the transfer function of LSTM units by $f_{LSTM}(∗)$.
> 其中 $I_t , F_t , O_t$ 分别表示输入门、遗忘门和输出门。$W_i , W_f , W_c, W_o$ 是它们的权重，$b_i , b_f , b_c, b_o$ 是偏置。$h_t$ 代表时间步 $t$ 的隐藏状态。记忆单元 $C_t$ 是由输入门调节的传入信息和由遗忘门 $F_t$ 调节的先前记忆的总和。为了简化，我们将 LSTM 单元的传递函数表示为 $f_{LSTM}(∗)$。
📌 **分析：**
* 对公式 (21) 中各个符号的详细解释，并引入了 $f_{LSTM}(∗)$ 作为 LSTM 单元的抽象表示。

---

> RNN can learn the statistical language model from a large number of normal texts, and then calculate the conditional probability distribution of the next word according to the previous generated words, and finally it can generate sentences that conform to such statistical language model.
> RNN 可以从大量正常文本中学习统计语言模型，然后根据前面生成的词计算下一个词的条件概率分布，最后生成符合该统计语言模型的句子。
📌 **分析：**
* 总结了 RNN（在此处特指使用 LSTM 单元的 RNN）作为解码器在语言生成方面的能力：学习语言模型，预测下一个词，并生成文本。

---

> Therefore, we can use a large number of normal texts to train the proposed model, let it learn the statistical language model and conditional probability distribution characteristics of normal texts.
> 因此，我们可以使用大量正常文本来训练所提出的模型，让它学习正常文本的统计语言模型和条件概率分布特征。
📌 **分析：**
* 强调了训练数据的重要性：通过大量正常文本的训练，模型能够捕获到自然语言的内在规律。

---

> After the training, when we use the trained model to generate sentences, we refer to Bowman et al. [47] and use the random sampled latent vector z as the initial state of RNN, then iteratively generate the following words.
> 训练完成后，当我们使用训练好的模型生成句子时，我们参考 Bowman 等人 [47] 的方法，使用随机采样的潜在向量 $z$ 作为 RNN 的初始状态，然后迭代生成后续词语。
📌 **分析：**
* **生成过程的关键**：
    * **“随机采样的潜在向量 $z$ 作为 RNN 的初始状态”**：这是 VAE 解码器与传统 RNN 生成器结合的核心点。编码器将正常文本映射到潜在空间，解码器则从潜在空间采样 $z$ 来生成。这个 $z$ 作为整个句子生成过程的全局信息，指导解码器生成具有特定风格或主题的句子，并确保其整体分布与正常文本相似。
    * **“迭代生成后续词语”**：标准的序列生成过程。

---

> For example, suppose currently we have alread generate i − 1 words and the model are going to generate the i -th word. The model first calculates the semantic information of all the previously generated $i −1$ words and stores in the hidden layer of the decoder, that is: $c_{i−1} = f_{L ST M} (x_1, x_2, . . . , x_{t−1}, z)$ (22)
> 例如，假设当前我们已经生成了 $i - 1$ 个词，模型将要生成第 $i$ 个词。模型首先计算所有先前生成的 $i-1$ 个词的语义信息，并存储在解码器的隐藏层中，即：
> $$c_{i−1} = f_{LSTM} (x_1, x_2, . . . , x_{i−1}, z) \quad \text{(22)}$$
📌 **分析：**
* **$c_{i-1}$**：代表解码器在生成第 $i$ 个词之前，结合了已生成的 $i-1$ 个词的上下文信息和来自 VAE 潜在空间 $z$ 的全局信息后，得到的隐藏状态或上下文向量。
* **$x_1, x_2, . . . , x_{i-1}$**：已生成的词序列。
* **$z$**：从潜在空间采样的向量，作为全局约束或“主题/风格”向量。
* 公式 (22) 描述了 LSTM 如何在每个时间步更新其内部状态，以捕捉上下文信息和潜在向量的影响。

---

> where $c_{i−1}$ represents the output vector of LSTM, which contains the semantic information of all the words generated previously.
> 其中 $c_{i−1}$ 代表 LSTM 的输出向量，它包含所有先前生成的词的语义信息。
📌 **分析：**
* 进一步解释了 $c_{i-1}$ 的作用。

---

> Based on these information, after all the hidden layers, we can calculate the probability distribution of the i -th word.
> 基于这些信息，经过所有隐藏层之后，我们可以计算第 $i$ 个词的概率分布。
📌 **分析：**
* 指出了从隐藏状态到词概率分布的下一步。

---

> To be more specific, we define the Prediction Weight (PW) as matrix $W_P ∈ R^{r×N}$, that is $W_P = \begin{bmatrix} w^p_{1,1} & w^p_{1,2} & \cdots & w^p_{1,N} \\ w^p_{2,1} & w^p_{2,2} & \cdots & w^p_{2,N} \\ \vdots & \vdots & \ddots & \vdots \\ w^p_{r,1} & w^p_{r,2} & \cdots & w^p_{r,N} \end{bmatrix}$ (23)
> 更具体地说，我们定义**预测权重（Prediction Weight, PW）**为矩阵 $W_P \in \mathbb{R}^{r \times N}$，即：
> $$W_P = \begin{bmatrix} w^p_{1,1} & w^p_{1,2} & \cdots & w^p_{1,N} \\ w^p_{2,1} & w^p_{2,2} & \cdots & w^p_{2,N} \\ \vdots & \vdots & \ddots & \vdots \\ w^p_{r,1} & w^p_{r,2} & \cdots & w^p_{r,N} \end{bmatrix} \quad \text{(23)}$$
📌 **分析：**
* **“预测权重 (PW)”**：这是一个可学习的矩阵，用于将 LSTM 隐藏层的输出（维度为 $r$）映射到词典大小 $N$ 的空间，从而为词典中的每个词生成一个分数。
* **$r$**：LSTM 最后一层隐藏层的单元数。
* **$N$**：词典的大小。

---

> where r indicates the number of LSTM units in the last hidden layer of decoder.
> 其中 $r$ 表示解码器最后一层隐藏层中 LSTM 单元的数量。
📌 **分析：**
* 对公式 (23) 中 $r$ 的解释。

---

> Then we use this learned matrix $W_P$ to calculate the score for each word in the dictionary D, that is $y_i = \sum_{k=1}^r w^p_{k,i} \cdot o^l_{i,t} + b^p_{i,t} , (24)$
> 然后我们使用这个学习到的矩阵 $W_P$ 来计算字典 $D$ 中每个词的分数，即：
> $$y_i = \sum_{k=1}^r w^p_{k,i} \cdot o^l_{i,t} + b^p_{i,t} \quad \text{(24)}$$
📌 **分析：**
* **$y_i$**：字典中第 $i$ 个词的分数（logits），在经过 Softmax 之前。
* **$o^l_{i,t}$**：LSTM 最后一层隐藏层在时间步 $t$ 的输出。
* **$w^p_{k,i}$**：预测权重矩阵 $W_P$ 中的元素。
* **$b^p_{i,t}$**：偏置项。
* 公式 (24) 描述了如何将 LSTM 的隐藏状态通过线性变换（乘以预测权重矩阵并加上偏置）映射到词典中每个词的未归一化分数。

---

> where $W_P$ and $b^p$ are learned weight matrix and bias. The dimension of the output vector y is N, which is the same size of dictionary D.
> 其中 $W_P$ 和 $b^p$ 是学习到的权重矩阵和偏置。输出向量 $y$ 的维度为 $N$，与字典 $D$ 的大小相同。
📌 **分析：**
* 对公式 (24) 中符号的进一步解释。

---

> In order to calculate the probability of next word at each step, we add a softmax classifier to the output layer to calculate the conditional probability of each word in dictionart D: $p(d_j | x_1, x_2, . . . , x_{t−1}, z) = \frac{exp(y_i )}{\sum_{j=1}^N exp(y_j )} , d_j ∈ D. (25)$
> 为了在每一步计算下一个词的概率，我们在输出层添加一个 **softmax 分类器（softmax classifier）**来计算字典 $D$ 中每个词的条件概率：
> $$p(d_j | x_1, x_2, . . . , x_{t−1}, z) = \frac{\exp(y_i )}{\sum_{j=1}^N \exp(y_j )} , d_j \in D. \quad \text{(25)}$$
📌 **分析：**
* **“softmax 分类器”**：将一组任意实数值（$y_i$，即 logits）转换为一个概率分布，其中所有概率的总和为1。
* **$p(d_j | \dots)$**：表示给定前面词的上下文信息和潜在向量 $z$ 时，下一个词是 $d_j$ 的条件概率。
* 公式 (25) 描述了如何从每个词的分数 $y_i$ 得到其在当前时间步的条件概率。

---

> For the task of automatic text generation, we only need to iteratively select the words with the highest conditional probability in the dictionary each time as the output, so that we can generate sentence with high quality.
> 对于自动文本生成任务，我们只需在每次迭代中选择字典中具有最高条件概率的词作为输出，这样我们就可以生成高质量的句子。
📌 **分析：**
* **普通文本生成**：解释了在没有信息隐藏需求时，如何生成高质量文本（贪婪地选择最高概率的词）。

---

> In fact, when the training sample size is large enough, there is actually a certain degree of redundancy in the conditional probability distribution space at each time step, so we can achieve secret information hiding by encoding it appropriately.
> 事实上，当训练样本规模足够大时，每个时间步的条件概率分布空间中实际上存在一定程度的**冗余（redundancy）**，因此我们可以通过适当编码来实现秘密信息隐藏。
📌 **分析：**
* **“冗余”**：这是隐写术存在的理论基础。在语言生成中，这意味着除了最高概率的词，还有许多其他概率不为零的词，它们同样能使句子保持自然。隐写术就是利用这些次优选择的空间来嵌入信息。

---

> For VAE-Stega (LSTM-LSTM), it shares the same model structure and the same decoder with VAE-Stega (BERT-LSTM), the only difference is the encoder module.
> 对于 **VAE-Stega (LSTM-LSTM)**，它与 **VAE-Stega (BERT-LSTM)** 共享相同的模型结构和解码器，唯一的区别在于编码器模块。
📌 **分析：**
* 回顾了 VAE-Stega 的两种版本，并指出它们的主要区别在于编码器。

---

> VAE-Stega (LSTM-LSTM) uses a recurrent neural network with LSTM units as encoder.
> VAE-Stega (LSTM-LSTM) 使用带有 LSTM 单元的循环神经网络作为编码器。
📌 **分析：**
* 明确了 VAE-Stega (LSTM-LSTM) 中编码器也使用了 LSTM。

---

> It uses the output of the last hidden layer at the last time step as the feature expression of the input sentence.
> 它使用最后一个时间步的最后一个隐藏层的输出作为输入语句的特征表达。
📌 **分析：**
* 描述了 LSTM 编码器如何生成句子的特征表示：通常取最后一个时间步的最终隐藏状态，因为它被认为是包含了整个序列信息。

---

> Therefore, for VAE-Stega (LSTM-LSTM), we only need to replace Formula (19) with the following form: $z(x) = f^l_{L ST M} (x ), (26)$
> 因此，对于 VAE-Stega (LSTM-LSTM)，我们只需用以下形式替换公式 (19)：
> $$z(x) = f^l_{LSTM} (x ), \quad \text{(26)}$$
📌 **分析：**
* 公式 (26) 表示 LSTM 作为编码器时，将输入句子 $x$ 映射为其特征表示 $z(x)$，其中 $f^l_{LSTM}$ 指的是 LSTM 模型的第 $l$ 层输出。

---

> where $l$ indicates the number of hidden layers in the encoder of VAE-Stega (LSTM-LSTM).
> 其中 $l$ 表示 VAE-Stega (LSTM-LSTM) 编码器中隐藏层的数量。
📌 **分析：**
* 对公式 (26) 中符号 $l$ 的解释。

---

> After extracting the feature expression of the input sentence, the follow-up operation is completely consistent with VAE-Stega (BERT-LSTM) as described above.
> 提取输入语句的特征表达后，后续操作与上述 VAE-Stega (BERT-LSTM) 的描述完全一致。
📌 **分析：**
* 强调两种 VAE-Stega 版本在编码器之后的流程是相同的，即都将编码器的输出转换为潜在分布的均值和方差，然后采样潜在向量并送入解码器。

---

> D. Information Hiding and Extraction
> D. **信息隐藏与提取**
📌 **分析：**
* 本小节将详细描述如何在 VAE-Stega 框架下具体实现秘密信息的嵌入和提取过程。

---

> As described in the previous part, in the process of automatic text generation, at each step, the decoder will calculate an N-dimensional conditional probability distribution vector, where each dimension represents the probability that the corresponding word in the dictionary D can be selected to be the output as the current step.
> 如前所述，在自动文本生成过程中，每一步，解码器都会计算一个 $N$ 维的**条件概率分布向量（conditional probability distribution vector）**，其中每个维度代表字典 $D$ 中相应词语被选作当前输出的概率。
📌 **分析：**
* 重申了解码器的输出：一个包含词典中所有词的概率的向量。

---

> The higher the probability of words, the more consistent with learned language model, so the better quality the text generated.
> 词语的概率越高，就越符合学习到的语言模型，因此生成的文本质量越好。
📌 **分析：**
* 这是基于语言模型生成文本的基本原则：选择高概率的词有助于保持文本的自然度和流畅性。

---

> When choosing different word as current output, the conditional probability distribution of subsequent words will be different.
> 当选择不同的词作为当前输出时，后续词语的条件概率分布将不同。
📌 **分析：**
* 这是序列生成的基本特性：每个词的选择都会影响后续的上下文和下一个词的概率分布。隐写术正是利用这种动态性来嵌入信息。

---

> At each time step for sentence generation, we first rank the words in the dictionary D in descending order according to their conditional probability, then choose the top m sorted words to build the Candidate Pool (CP).
> 在句子生成的每个时间步，我们首先根据条件概率将字典 $D$ 中的词按降序排列，然后选择前 $m$ 个词来构建**候选池（Candidate Pool, CP）**。
📌 **分析：**
* **“候选池（CP）”**：信息嵌入的关键。不是从整个词典中选择词，而是从一个缩小范围的、概率较高的词集合中选择，以尽可能保持文本质量。
* **“降序排列”**：确保高概率词优先被考虑。

---

> The process of information embedding is to encode the candidate pool at each moment, and then output corresponding candidate word according to the secret information to be embedded, so that the generated text contains secret information.
> 信息嵌入的过程是：在每个时刻对候选池进行编码，然后根据要嵌入的秘密信息输出相应的候选词，从而使生成的文本包含秘密信息。
📌 **分析：**
* **信息嵌入原理**：通过在候选池中根据秘密比特位来选择词，秘密信息就被编码到生成的文本中。例如，如果候选池有4个词，而秘密比特是“00”，就选择第一个词；如果是“01”，选择第二个词，以此类推。

---

> Generally speaking, the larger the candidate pool is, the more bits can be embedded, but at the same time, it is more likely to select words with lower conditional probability at the current output and thus decrease the quality of generated sentence.
> 一般来说，候选池越大，可以嵌入的比特数就越多，但同时，当前输出选择条件概率较低词的可能性也越大，从而降低生成语句的质量。
📌 **分析：**
* **嵌入率与质量的权衡**：
    * **大候选池**：提供更多选择，可以嵌入更多信息（高嵌入率）。
    * **质量下降**：但大候选池也意味着可能选择“不那么自然”的词（概率较低的词），从而影响文本质量。
* 这体现了隐写术中常见的嵌入率与不可察觉性之间的权衡关系。

---

> In this paper, we refer to previous works and compare two different encoding methods, namely Huffman coding (HC) [13] and arithmetic coding (AC) [17], to illustrate the effectiveness of the proposed VAE-Stega model.
> 在本文中，我们参考了以往的工作，并比较了两种不同的编码方法，即**霍夫曼编码（Huffman coding, HC）** [13] 和**算术编码（arithmetic coding, AC）** [17]，以说明所提出的 VAE-Stega 模型的有效性。
📌 **分析：**
* **“霍夫曼编码（HC）”** [13]：一种变长编码方法，根据符号的频率（在这里是词的条件概率）为高频词分配短编码，为低频词分配长编码。
* **“算术编码（AC）”** [17]：一种更高级的变长编码方法，它可以将整个消息编码为一个小数，并可以实现接近信息熵的压缩比，通常比霍夫曼编码更灵活，能更好地利用概率分布。
* 选择这两种编码方法进行比较，以评估 VAE-Stega 模型在不同嵌入策略下的性能。

---

> The process of secret information hiding and extracting is a completely opposite process.
> 秘密信息隐藏和提取的过程是完全相反的过程。
📌 **分析：**
* 隐写术的普遍原则：提取是嵌入的逆操作，需要相同的算法和密钥。

---

> It is worth noting that, compared to previous steganographic text generation algorithms, such as the RNN-Stega model [13], the proposed VAE-Stega model needs to use the same latent vector sampled from the latent space as the sender in the process of decoding each steganographic sentence.
> 值得注意的是，与以前的隐写文本生成算法（例如 RNN-Stega 模型）相比，所提出的 VAE-Stega 模型在解码每个隐写语句的过程中需要使用与发送方**相同的从潜在空间中采样的潜在向量 $z$**。
📌 **分析：**
* **VAE-Stega 的特殊要求**：这是 VAE-Stega 框架下信息提取的关键难点。传统 RNN-Stega 只需要共享语言模型和编码方法，但 VAE-Stega 还要求发送方和接收方在生成（和提取）时使用相同的潜在向量 $z$。

---

> In actual use, since the computer can only generate pseudo-random numbers, the sender and receiver only need to share the random seed generation algorithm to ensure that the sampled latent vectors are synchronized, so as to ensure correct extraction secret information in the received steganographic text.
> 在实际使用中，由于计算机只能生成**伪随机数（pseudo-random numbers）**，发送方和接收方只需共享**随机种子生成算法（random seed generation algorithm）**，以确保采样的潜在向量同步，从而保证在接收到的隐写文本中正确提取秘密信息。
📌 **分析：**
* **解决方案**：利用计算机生成伪随机数的特性。只要发送方和接收方使用相同的随机数生成器（以及相同的随机种子），它们就能生成相同的潜在向量 $z$，从而实现同步。这类似于密码学中共享密钥的原理。

---

> At each time step, Bob inputs each word into the same trained model and gets the conditional probability distribution of next word.
> 在每个时间步，鲍勃（Bob）将每个词输入到相同的训练模型中，并获得下一个词的条件概率分布。
📌 **分析：**
* 提取过程的第一步：接收方（Bob）使用与发送方相同的训练好的模型来预测下一个词的条件概率分布。

---

> He firstly sorts all the words in the dictionary in descending order of probability and selects the top m words to form the Candidate Pool.
> 他首先根据概率将字典中所有词按降序排列，并选择前 $m$ 个词来形成**候选池（Candidate Pool）**。
📌 **分析：**
* 提取过程的第二步：重建候选池，这与嵌入时的步骤相同。

---

> Then he encodes the candidate pool using the same encoding method as the sender, such as Huffman coding [13] or arithmetic coding [17].
> 然后他使用与发送方相同的编码方法（例如霍夫曼编码 或算术编码）来编码候选池。
📌 **分析：**
* 提取过程的第三步：使用相同的编码方法重建编码规则。

---

> Finally, according to the actual transmitted word at the current moment, the reciever can successfully and accurately decode the bits embedded in each word, thus to complete the covert communication.
> 最后，根据当前实际传输的词，接收方可以成功且准确地解码嵌入在每个词中的比特，从而完成隐蔽通信。
📌 **分析：**
* 提取过程的最后一步：通过将接收到的实际词与重建的编码规则进行匹配，反推出嵌入的秘密比特。
* 总结了整个隐蔽通信的闭环过程。