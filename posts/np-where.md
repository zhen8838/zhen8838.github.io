---
title: python 多维数组指定区域中寻找元素索引
categories:
  - 编程语言
date: 2019-01-06 18:57:12
tags:
-   Python
---

今天想完成一个功能,需要在**一个多维数组中的指定区域找到对应元素在整个数组中的索引**,因为这个问题描述起来不方便,找了半天也没有找到好的答案.因此就自己尝试了一下`np.where`果然可以,但是网上的一些例子中都没提到这个的用法,所以记录一下.

<!--more-->

## CODE
```python
import numpy as np


if __name__ == "__main__":
    # 构造数组
    a = np.zeros((7, 10, 5, 25))
    # 首先自己设置几个值
    dim1 = np.random.randint(0, high=7, size=3)
    dim2 = np.random.randint(0, high=10, size=3)
    dim3 = np.random.randint(0, high=5, size=3)
    a[dim1, dim2, dim3, 4] = 1
    # 我需要在最后一维的第5列找到大于0.7的元素的索引
    idex1, idex2, idex3 = np.where(a[..., 4] > .7)
    print('dim1:',dim1, idex1)
    print('dim2:',dim2, idex2)
    print('dim3:',dim3, idex3)

```

## RUN
```sh
dim1: [6 1 6] [1 6 6]
dim2: [1 3 1] [3 1 1]
dim3: [1 4 4] [4 1 4]
```

