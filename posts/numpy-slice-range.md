---
title: numpy中动态范围切片
categories:
  - 编程语言
date: 2019-06-08 20:11:51
tags:
-   Python
-   Numpy
---
今天想把两个不同的形状的数组进行赋值,因为数组形状是动态的,所以要想一个办法进行动态的范围切片.

<!--more-->

# 解决方案

两个数组:

    A = [[3,3,32,3],[32]]
    B = [[3,3,64,3],[64]]
    赋值 B -> A

```python
import numpy as np
A = np.array([np.random.rand(3, 3, 32, 3), np.random.rand(32)])
B = np.array([np.random.rand(3, 3, 64, 3), np.random.rand(64)])
for i in range(len(A)):
    A[i] = B[i][[slice(0, s) for s in A[i].shape]]

assert np.array_equal(A[0], B[0][:, :, :32, :])
assert np.array_equal(A[1], B[1][:32])

```