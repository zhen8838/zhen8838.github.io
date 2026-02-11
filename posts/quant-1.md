---
title: 神经网络量化-基本原理
mathjax: true
toc: true
categories:
  - 编译器
date: 2021-03-27 09:44:40
tags:
- 神经网络量化
---

准备系统的学习一下神经网络量化，参考网络上的一些教程同时再次整理消化，这次首先对基本原理进行了解。

参考资料： [jermmyxu的神经网络量化系列教程](https://www.cnblogs.com/jermmyhsu/p/13169254.html)


<!--more-->

```python
import numpy as np
from torchvision.models.mobilenet import mobilenet_v2
from torch import nn 
import matplotlib.pyplot as plt
np.set_printoptions(precision=4,suppress=True)
```

## 网络量化的基本原理

### 背景知识

我们通常会将一张 uint8 类型、数值范围在 $0\sim255$ 的图片归一成 float32 类型、数值范围在 $0.0\sim1.0$ 的张量，这个过程就是反量化。

最简单的量化公式是min_max_map，假设使用这里我们用$r$表示浮点实数，$q$表示量化后的定点整数。浮点和整型之间的换算公式为：
$$q = round(\frac{r}{S}+Z)$$
$$r = S(q-Z)$$

其中，$S$ 是 scale，表示实数和整数之间的比例关系，$Z$ 是 zero point，表示实数中的 0 经过量化后对应的整数，它们的计算方法为：
$$S = \frac{r_{max}-r_{min}}{q_{max}-q_{min}}$$
$$Z = round(q_{max} - \frac{r_{max}}{S})$$

首先我先找一个mobilenet，得到符合真实场景的参数均值和方差。


```python
mbv2 = mobilenet_v2(pretrained=True)
conv2d: nn.Conv2d = mbv2.features[13].conv[0][0]
R= conv2d.weight[:,:,0,0].detach().numpy()
R.mean(), R.std()
```




    (0.00031194088, 0.055000253)




```python
r = np.random.normal(loc=0.00031194088,scale=0.055000253,size=10)
def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = float((max_val - min_val) / (qmax - qmin))
    zero_point = qmax - max_val / scale
    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax
    zero_point = int(zero_point)
    return scale, zero_point
def quantFloat(r,scale, zero_point, num_bits=8):
    return np.round(np.clip(r/scale + zero_point,0,2**num_bits-1)).astype('int')
S,Z=calcScaleZeroPoint(r.min(),r.max())
q=quantFloat(r,S,Z)
print('r:',r,'\nS:',S,'\nZ:',Z,'\nq:',q)
```

    r: [ 0.025   0.0205 -0.0064 -0.0634  0.0538  0.0205  0.0099  0.0321 -0.0302
     -0.0709] 
    S: 0.0004889321179676555 
    Z: 145 
    q: [196 187 132  15 255 187 165 211  83   0]


可以发现反量化的时候将出现一定的误差：


```python
def dequantUint8(q, scale, zero_point):
    return scale * (q - zero_point)
re_r=dequantUint8(q,S,Z)
print(r-re_r)
```

    [ 0.     -0.     -0.      0.0002 -0.     -0.0001  0.0002 -0.0002  0.0001
     -0.    ]


### 矩阵运算的量化
假设$r_1,r_2$是两个矩阵,他们的维度分别为$N\times M,M\times K$,$r_3$作为两个矩阵的乘积结果,维度是$N \times K$,乘积计算过程如下:

$$
r_3^{n,k}=\sum_{m=1}^M r_1^{n,m}r_2^{m,k}
$$

设$S_1,Z_1$对应$r_1$的量化因子,同理得到$S_2,Z_2$和$S_3,Z_3$,这时候将上述矩阵相乘公式的量化计算过程如下:
$$
\begin{aligned}
S_3(q_3^{n,k}-Z_3)&=\sum_{m=1}^{M}S_1(q_{1}^{n,m}-Z_1)S_2(q_2^{m,k}-Z_2)\\
S_3q_3^{n,k}&=S_1S_2 \sum_{m=1}^{M}(q_{1}^{n,m}-Z_1)(q_2^{m,k}-Z_2)+S_3 Z_3\\
q_3^{n,k}&=\frac{S_1S_2}{S_3} \sum_{m=1}^{M}(q_{1}^{n,m}-Z_1)(q_2^{m,k}-Z_2)+Z_3
\end{aligned}
$$
此时上述公式中只有$\frac{S_1S_2}{S_3}$部分是浮点运算(并且这个浮点数被大量实验证明了位于0-1之间),假设未经过重新量化的矩阵计算结果为$Q$,固定浮点数$\frac{S_1S_2}{S_3}$定义为$F$,那么对于一个浮点数可以利用一个技巧进行近似,然后可以将这个浮点计算转换为定点计算.即将这个浮点数用一个定点整数$F_0$*定点小数$2^{-n}$的方法来近似:
$$
\begin{aligned}
q_3^{n,k}&=F Q+Z_3\\
&=2^{-n}F_0 Q+Z_3
\end{aligned}
$$


```python
def get_r_q_mat(h,w,loc=0.001,scale=0.004):
    r = np.random.normal(loc=loc,scale=scale,size=(h,w))
    s,z=calcScaleZeroPoint(r.min(),r.max())
    q = quantFloat(r,s,z)
    return r,q,s,z
r_1,q_1,S_1,Z_1= get_r_q_mat(3,5)
r_2,q_2,S_2,Z_2= get_r_q_mat(5,4)
r_3 = r_1 @ r_2
S_3,Z_3=calcScaleZeroPoint(r_3.min(),r_3.max())
q_3 = quantFloat(r_3,S_3,Z_3)
Q = (q_1-Z_1) @ (q_2-Z_2)
F= (S_1*S_2)/S_3
print(F)
```

    0.008822082202930992


此时我们计算$F$的定点数$F_0$和$2^{-n}$,其实就是计算哪一个$F_0$和$n$近似$F$的误差最小:
$$
\begin{aligned}
    \text{argmin}_{n}&abs(FQ-(2^{-n} \times F_0)Q,\ \  F_0 \in [q_{min},q_{max}]\\
    F_0 &= round(\frac{F}{2^{-n}}) = round(F * (1 << n)) \\
    将F_0展开,并将2^{-n}转换为位运算&: \\
    \text{argmin}_{n}&abs(FQ-(round(F * (1 << n))Q >> n))
\end{aligned}
$$
接下来给出一个确定$F_0$与$n$的代码:


```python
def get_f_0(F,Q,is_print=True):
    mind=1e9
    minn=-1
    for n in range(0,16):
        F_0=int(round(F * (1<<n)))
        diff = abs(F*Q - (int(F_0*Q)>>n))
        if diff<mind:
            mind,minn=diff,n
        if is_print:
            print(f"n={n},F_0={F_0},diff={diff}")
    return int(round(F * (1<<minn))),minn
F_0,n = get_f_0(F,int(Q[0][0]))
```

    n=0,F_0=0,diff=175.27712920783293
    n=1,F_0=0,diff=175.27712920783293
    n=2,F_0=0,diff=175.27712920783293
    n=3,F_0=0,diff=175.27712920783293
    n=4,F_0=0,diff=175.27712920783293
    n=5,F_0=0,diff=175.27712920783293
    n=6,F_0=1,diff=135.72287079216707
    n=7,F_0=1,diff=19.277129207832928
    n=8,F_0=2,diff=19.277129207832928
    n=9,F_0=5,diff=19.722870792167072
    n=10,F_0=9,diff=0.27712920783292816
    n=11,F_0=18,diff=0.27712920783292816
    n=12,F_0=36,diff=0.27712920783292816
    n=13,F_0=72,diff=0.27712920783292816
    n=14,F_0=145,diff=0.7228707921670718
    n=15,F_0=289,diff=0.7228707921670718


观察上述输出,可以发现在$n=10$的时候就可以得到较好的量化因子了,这样所有的运算就可以用定点的方式来计算了.
 
此时我们写出一个完整的例子，分别使用$q_3$的两种解量化方式来检查量化运算的精度损失有多大：

$$
\begin{aligned}
\text{deq\_r}_{3}^1 &= dequant(q_3) \\

\text{deq\_r}_{3}^2 &= dequant(F Q + Z_3) \\

\text{deq\_r}_{3}^3 &= dequant(2^{-n} F_0 Q + Z_3)

\end{aligned}
$$


```python
deq_r_3_1=dequantUint8(q_3,S_3,Z_3)
deq_r_3_2=dequantUint8(F*Q+Z_3,S_3,Z_3)-r_3
deq_r_3_3=dequantUint8(np.right_shift(F_0*Q,n)+Z_3,S_3,Z_3)
print((deq_r_3_1-r_3).max())
print((deq_r_3_2-r_3).max())
print((deq_r_3_3-r_3).max())
```

    1.6667435495922793e-07
    4.1159863929325536e-05
    3.4088182772356416e-07


```python
可以发现精度损失并不大，表明了当前的方法有用。
```
