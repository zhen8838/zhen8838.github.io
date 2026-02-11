---
title: Tensorflow 1.15中TensorBoard错误
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-10-21 14:01:22
tags:
- Tensorflow
---

升级了`tensorflow`到1.15,发现一个用`tf.keras`中`TensorBoard`的时候就会报错的问题.

<!--more-->

# 问题描述

错误信息如下:
```sh
TypeError: An op outside of the function building code is being passed
a "Graph" tensor. It is possible to have Graph tensors
leak out of the function building context by including a
tf.init_scope in your function building code.
For example, the following function will fail:
  @tf.function
  def has_init_scope():
    my_constant = tf.constant(1.)
    with tf.init_scope():
      added = my_constant * 2
The graph tensor has name: create_file_writer/SummaryWriter:0
```

# 问题解决

这个问题主要还是因为`tensorflow`要兼容两个版本的问题,也有我用`vscode`开发的原因.

现在使用的`tensorflow.keras`的时候最好还是用如下的方式:
```python
TensorBoard = tf.keras.callbacks.TensorBoard
```

我用的是如下:
```python
from tensorflow.python.keras.callbacks import TensorBoard
```

问题就在这里了,现在`tensorflow 1.15`默认的`TensorBoard`是使用`tf 2.0`的写法的,如果用第一个方式调用是没问题的.但因为`tf.keras.callbacks.TensorBoard`是从`from tensorflow.python.keras.callbacks_v1`导入的,所以我的写法调用的是对应`tf 2.0`的代码,导致错误.

所以我的写法要改成:
```python
from tensorflow.python.keras.callbacks_v1 import TensorBoard
```