## 概率重组模块
该模块描述了一类方案，其功能是将任意离散分布的概率值进行**分解（decompose）**和**重组（recombine）**，形成一组互不重叠的**均匀分布（uniform distributions）**。
例如，如果一个词元的概率是 0.5，它可以被分解为 0.4 和 0.1 两个块。
在将所有概率都切成“块”之后，下一步就是把这些零散的块重新组合成一个个的“箱子”。但组合不是随意的，必须遵循严格的规则。
- 每个块必须且只能被分配到一个箱子中。
- 同一个箱子内的所有块必须具有相等的概率值。
- 任何一个词元在同一个箱子中不能拥有一个以上的块。
## 分箱采样
根据**PRG**选择由概率重组模块输出的箱子
## 均匀隐写模块
根据**密文**从分箱采样得到的箱子中选择对应的token
## 安全性证明

## 框架的典型实现
作者根据自己的框架提出了
- 基于差分的重组
- 基于二进制的重组
- 基于稳定性的重组
- 循环移位均匀隐写（均匀隐写模块）
- **如何把discop纳入自己的框架中**
## 框架内的最优方案
## 实验



