---
title: 统计学习方法:KNN
mathjax: true
toc: true
categories:
  - 机器学习
date: 2020-05-23 20:51:46
tags:
- 统计学习方法
- 概率论
---

`K近邻法`比较简单,我就讲下流程.

<!--more-->

1.  确定距离度量

在$L_p$距离中选择任意一个即可

$$
\begin{aligned}
  L_{p}\left(x_{i}, x_{j}\right)=\left(\sum_{l=1}^{n}\left|x_{i}^{(l)}-x_{j}^{(l)}\right|^{p}\right)^{\frac{1}{p}}
\end{aligned}
$$


2.  计算待分类样本点与已知样本点的距离

3.  根据$k$值确定待分类样本点类别