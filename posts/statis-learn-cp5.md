---
title: 统计学习方法:逻辑回归
mathjax: true
toc: true
categories:
  - 机器学习
date: 2020-05-29 16:47:27
tags:
- 统计学习方法
- 概率论
---

这一章其实是逻辑回归和最大熵模型,最大熵模型的实现需要数个特征与定义对应的特征函数,因此我暂时没有实现.

<!--more-->


# 逻辑回归

## 原理

假设是一个二分类问题,我们把分类问题考虑为由样本$X$得到对应$Y$的概率,那么模型可以被定义为$P(Y|X)$.为了将输出分配为概率形式,使用`logistic`分布:
$$
\begin{aligned}
  F(x)=P(X\leq x)=\frac{1}{1+e^{-\theta^T x}}
\end{aligned}
$$

那么对于一个样本计算他类别为`1`的概率的过程可以记做$p=\sigma(\theta^Tx)$,此时我们面对的是二分类问题,那么自然类别为`0`的概率为$1-p$得到对应的概率质量函数为:
$$
\begin{aligned}
  \text{Let} \ \ \ p&=\sigma(\theta^Tx)\\
  P(y)&=\begin{cases}p&,y=1\\1-p&,y=0  \end{cases}\\
      &= y^{p}+(1-y)^{1-p}\\
  将P(y)表示为&P(y|x;\theta)
\end{aligned}
$$

有了对应的概率质量函数,我们需要求解最合适的参数$\theta$,就可以利用最大似然估计的方法,最大似然考虑到了分布函数的联合分布,当他们的联合分布概率最大时,那么$\theta$肯定是最优的.:
$$
\begin{aligned}
L(x_1,x_2,...,x_n;\theta)&=\prod_{i=1}^n P(y|x_i;\theta)\\
&=\prod_{i=1}^n\left[ \sigma(\theta^Tx_i)^{y_i} +(1-\sigma(\theta^Tx_i))^{1-y_i}\right]\\
对数化&\\
L(x_1,x_2,...,x_n;\theta)&=\log\left[ \prod_{i=1}^n\left[ \sigma(\theta^Tx_i)^{y_i} +(1-\sigma(\theta^Tx_i))^{1-y_i}\right]\right]\\
&=\sum_{i=1}^n\left[ y_i \log\sigma(\theta^Tx_i) +(1-y_i)\log(1-\sigma(\theta^Tx_i))\right]\\
&=\sum_{i=1}^n\left[ y_i \log\frac{1}{1+e^{-\theta^Tx_i}}  +(1-y_i)\log(1-\frac{1}{1+e^{-\theta^Tx_i}})\right]\\
&=\sum_{i=1}^n\left[ y_i (\theta^Tx_i) - \log(1+e^{\theta^Tx_i})\right]
\end{aligned}
$$

注意到极大似然估计对数化后的第二步实际上就等价于负的交叉熵,所以令似然估计最大化,相当于最小化交叉熵,因此模型的损失即为:
$$
\begin{aligned}
  \mathcal{L}=-\sum_{i=1}^n\left[ y_i (\theta^Tx_i) - \log(1+e^{\theta^Tx_i})\right]
\end{aligned}
$$

有了损失,又可以求导.可以采用梯度下降法进行优化,梯度为如下.
$$
\begin{aligned}
  \frac{d\mathcal{L}}{d\theta} &=-(yx-\frac{e^{\theta^Tx}\cdot x}{1+e^{\theta^Tx}})\\
  &=-(y-\frac{1}{1+e^{-w^Tx}})x
\end{aligned}
$$


# 最大熵模型


最大熵模型很难给出具体的例子,我看书也看的有点晕,后面参考到苏神的文章才理解一些.总体来说是求解满足各种约束条件下包含最大条件熵$H(P(Y|X))$的模型.