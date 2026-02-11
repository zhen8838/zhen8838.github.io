---
title: CapsNet实现以及踩坑
categories:
-   深度学习
date: 2019-04-22 15:26:49
mathjax: true
tags:
-   Tensorflow
---

我本来打算用`tensorflow 2.0`去写`capsule net`的,结果被`tf 2.0`中的`tensorboard`坑的放弃了...

然后我换成`tf 1.13`去写了,下面做一个实现过程记录.

<!--more-->

# 前言

其实我感觉`capsule net`应该是全连接层的升级版,以前的全连接层矩阵输入为$x=[batch,m]$,现在`capsule net`的全连接层为$x=[batch,m,len]$,就是将以前的一维特征扩展成了向量特征.

# 数据流动过程

其实我一开始的时候一直想不通数据流动时候的矩阵维度,所以卡壳了,下面就讲一下数据流动过程.

参考的网络结构是:
![](capsnet/1.png)

其实从`reshape`开始才是`capsule net`,1152为向量个数,8为向量长度.下一层连接到10个长为16的向量.最后计算向量的长度范数来匹配`minsit`的分类节点.

然后在`capsDense`中使用`routing`算法来更新`c`的值,最后输出.

# 踩过的坑

## 1. squash

### 问题描述

激活函数公式为:

$$ \begin{aligned}
    v_j=\frac{||s_j||^2}{1+||s_j||^2}\frac{s_j}{||s_j||}
\end{aligned} $$

下面`my_squash`是我写的:
```python
def my_squash(s):
    s_norm = tf.norm_v2(s)
    s_square_norm = tf.square(s_norm)
    v = (s_square_norm * s)/((1+s_square_norm)*s_norm)
    return v
```
然后我训练的时候一直没有效果,我找了半天才明白.

### 问题解决

问题在于`tf.norm_v2`这里的维度控制,他这里的范数指的是每一个向量的长度范数,所以需要指定维度为`s_norm = tf.norm_v2(s, axis=-1, keepdims=True)`,下面是正确的维度演示:
$$ \begin{aligned}
    令 S_j &= [batch,1152,8] \\
    则 ||S_j|| &= [batch,1152,1] \\
    ||S_j||^2 &= [batch,1152,1] \\
    v&= [batch,1152,8]
\end{aligned} $$

并且这个函数其实可以优化为:
$$ \begin{aligned}
    v_j&=\frac{||s_j||^2}{1+||s_j||^2}\frac{s_j}{||s_j||}\\
    &=\frac{||s_j||s_j}{1+||s_j||^2}
\end{aligned} $$
代码为:
```python
with tf.variable_scope('squash'):
    s_norm = tf.norm_v2(s, axis=-1, keepdims=True)
    s_square_norm = tf.square(s_norm)
    v = (s_norm * s)/(1+s_square_norm)
    return v
```