---
title: 统计学习方法:感知机
mathjax: true
toc: true
categories:
  - 机器学习
date: 2020-05-19 09:53:37
tags:
- 统计学习方法
- 概率论
---

我觉得自己对于概率视角下的机器学习方法还是不够清晰,因此开个新坑(其实这个基础就应该上上学期打好),现在准备两个月之内把统计学习方法第二版撸完(~~flag是不是太...~~).不管了,今天是第一章感知机,为了节约记录的时间,我都只写我觉得比较重要的地方.

<!--more-->
# 一般形式

## 模型

$$
\begin{aligned}
  f(x)=sign(w\cdot x+b)\\
  sign(x)=\begin{cases}+1,x\geq0\\-1,x< 0 \end{cases}
\end{aligned}
$$



## 学习策略

因为是要对所有数据点进行二分类,所以直观的想法是根据误分类点到决策面$S$的距离作为损失:
$$
\begin{aligned}
  -\frac{1}{\parallel w\parallel}y_i(w\cdot x_i+b)
\end{aligned}
$$

因为感知机想法比较简单,不需要考虑决策面距离分类点有多远,所以舍去$\frac{1}{\parallel w\parallel}$得到损失函数:

$$
\begin{aligned}
  -y_i(w\cdot x_i+b)
\end{aligned}
$$

## 训练

感知机虽然是通过求导反向传播更新的,但是要注意直接用`batch`的方式学习是没有用的!这里我踩了个坑,他的更新方式是选择分类错误的点的`loss`进行反向传播.

1.  定义初值$w_0,b_0$

2.  输入$x_i,y_i$

3.  如果$-y_i(w\cdot x_i +b)>0$

$$
\begin{aligned}
  w\leftarrow w+\eta y_i x_i\\
  b\leftarrow b+\eta y_i 
\end{aligned}
$$

4.  重复2,3

# 对偶形式

## 模型
我们注意到$w$在经过多次更新后,他的增量实际上等于如下:
$$
\begin{aligned}
  \text{Let}\ \ \ \ \alpha_i= n_i \eta\\
  w=\sum_{i=1}^N \alpha_i y_i x_i\\
  b=\sum_{i=1}^N \alpha_i y_i \\
\end{aligned}
$$

将原始感知机中的替换为如下:
$$
\begin{aligned}
  f(x)=sign(\sum_{j=1}^N \alpha_j y_j x_j \cdot x +b)
\end{aligned}
$$

## 训练


1.  模型定义

$$
\begin{aligned}
  f(x)=sign(\sum_{j=1}^N \alpha_j y_j x_j \cdot x +b)\\
  \alpha=(\alpha_1,\alpha_2,...\alpha_N)^T
\end{aligned}
$$

2.  初始化参数$\alpha,b$

3.  输入$x_i,y_i$

**NOTE** 这里为了加速计算,首先将所有的$\sum_{j=1}^N \sum_{i=1}^N x_j \cdot x_i$(称为`Gram矩阵`)计算出来,训练的时候直接取值即可:
$$
\begin{aligned}
  \boldsymbol{G}=[x_i\cdot x_j]_{N\times N}
\end{aligned}
$$

4.  如果$-y_i(\sum_{j=1}^N \alpha_j y_j x_j \cdot x_i +b)>0$


$$
\begin{aligned}
  \alpha_i\leftarrow \alpha_i+\eta\\
  b\leftarrow b+\eta y_i 
\end{aligned}
$$

5.  重复3,4