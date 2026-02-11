---
title: 对比tensordot、matmul、einsum速度
categories:
  - 深度学习
date: 2019-04-20 18:28:12
tags:
-   Tensorflow
---

准备自己实现`capsule Net`，今天看了下别人实现的版本，感觉里面的矩阵乘积应该是可以优化的。

然后我写代码的时候，感觉一个可以优化的点是不同维度之间的`Tensor`的矩阵乘积，所以我做了一个小测试。

<!--more-->

# 说明

因为`capsule net`中全连接需要权值乘上输入向量：
$$ 
\begin{aligned}
    \hat{u}_{j|i}&=W_{ij}u_i \\
    W_{ij} &= [Len_{l},Len_{l+1}] \\
    u_i &= [batch,N_l,Len_{l}]
\end{aligned}
$$

他的实例是:
$$ 
\begin{aligned}
    W_{ij} &= [8,16] \\
    u_i &= [batch,1152,8]
\end{aligned}
$$

因为两个`Tensor`的维度不一样,所以在他的代码中都是`tile`然后进行计算的.然后我找了几个矩阵计算的函数进行比较(使用 tensorflow 2.0).

```python
import tensorflow.python as tf
import numpy as np
import os
import timeit


# @tf.function
def test_tensordot(W: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
    v = tf.tensordot(u, W, axes=[[2], [0]])
    return v


# @tf.function
def test_matmul(W: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
    W_ = W[tf.newaxis, tf.newaxis, ...]
    u_ = u[..., tf.newaxis]
    W_ = tf.tile(W_, [u.shape[0], 1152, 1, 1])
    v = tf.matmul(W_, u_, transpose_a=True)
    return tf.squeeze(v)


# @tf.function
def test_einsum(W: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
    return tf.einsum('ij,aki->akj', W, u)


def test_compare():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    batch = 16
    tf.set_random_seed(1)
    W = tf.get_variable('W', shape=(8, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())
    u = tf.get_variable('u', shape=(batch, 1152, 8), dtype=tf.float32, initializer=tf.initializers.random_normal())

    start = timeit.default_timer()
    for i in range(100):
        v1 = test_tensordot(W, u)
    tim = timeit.default_timer()-start
    print("tensordot", tim)

    start = timeit.default_timer()
    for i in range(100):
        v2 = test_matmul(W, u)
    tim = timeit.default_timer()-start
    print("matmul", tim)

    start = timeit.default_timer()
    for i in range(100):
        v3 = test_einsum(W, u)
    tim = timeit.default_timer()-start
    print("einsum", tim)

    print(np.allclose(v1, v2, atol=0.5e-6))
    print(np.allclose(v1, v3, atol=0.5e-6))


test_compare()
```

# 结果

```sh
(tf2) ➜  tf2 /home/zqh/miniconda3/envs/tf2/bin/python /home/zqh/Documents/tf2/test/test_fuc.py
tensordot 0.2818375900023966
matmul 0.09134677500696853
einsum 0.051768514000286814
True
True
```

实验发现`einsum`的效率更加高.


# 疑问
在`tensorflow 2.0`中明明可以使用`@tf.function`来优化运行速度.但是我在上面的程序中使用这个方式,反而速度更慢了...

```sh
(tf2) ➜  tf2 /home/zqh/miniconda3/envs/tf2/bin/python /home/zqh/Documents/tf2/test/test_fuc.py 
# 不使用 @tf.function
tensordot 0.21580070699565113
matmul 0.08182674000272527
einsum 0.044429186993511394
True
True
(tf2) ➜  tf2 /home/zqh/miniconda3/envs/tf2/bin/python /home/zqh/Documents/tf2/test/test_fuc.py
# 使用 @tf.function
tensordot 0.27514774599694647
matmul 0.15171915300015826
einsum 0.0524767349998001
True
True
```