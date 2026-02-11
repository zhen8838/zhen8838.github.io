---
title: 基于DL的CostModel
mathjax: true
toc: true
categories:
  - 编译器
date: 2023-06-16 11:38:49
tags:
- CostModel
- 后端优化
---

调研一些使用机器学习/深度学习方法构造神经网络CostModel的论文.

<!--more-->


# [TLP: A Deep Learning-based Cost Model for Tensor Program Tuning](https://github.com/zhaiyi000/tlp)

他这里是把对源代码的schedule的类型进行onehot, 然后名字参数进行 tokenize, 数值参数不改变.

$$
\begin{aligned}
F =  F_{1} (\tau) (F_{2} (id) |F_{3} (num)) \\
F_1 : \text{PrimitiveType} \rightarrow \text{OnehotVector}  \\
F_2 : \text{NameParam} \rightarrow \text{Token}  \\
F_3 : \text{Number} \rightarrow \text{Number}  \\
\text{PrimitiveType} \in { \text{split}, \text{reorder}, \text{fuse} } \\
\text{NameParam} := \text{id}
\end{aligned}
$$


特征提取的流程图如下: 

![](dl-costmodel/TLP.png)

他的模型基本上是基于transformer, 讲数据加载进来之后分为`input[:setp_size,:feat_size]`, 这里`setp_size,feat_size`分别为25,22. 应该说默认一共调度25次, 以及每个调度的参数长22.

# [Efficient Automatic Scheduling of Imaging and Vision Pipelines for the GPU](https://aekul.github.io/gpu_autoscheduler/)

这个是通过分析原始调度中的一系列特征值进行分类. 将`pipeline_features, schedule_features`送到两个输入头中, 然后分别进行全连接之后再concat之后继续全连接.
