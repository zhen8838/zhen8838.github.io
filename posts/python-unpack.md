---
title: python返回值进行unpack
categories:
  - 编程语言
date: 2019-05-09 10:02:34
tags:
- Python
---

最近在写`yolov3`,因为`yolov3`的多输出性质,所以我打算写适配多输出的工具函数,在`numpy`中可以在一个`array`中包含多个不同维度的`array`,但在`tensorflow`中一个`tensor`只能保存相同维度的矩阵,这就十分蛋疼了.下面记录一下我是如何解决的.

<!--more-->

# 问题描述

在做`parser`的时候,让其返回值第一个为`img`,然后是一个动态的`label`数组,接下来使用`tensorflow`的包装函数进行包装,最后执行:
```python
def t():
    img = np.zeros((240, 320, 3))
    labels = [np.array((7, 10, 3, 25)), np.array((14, 20, 3, 25))]
    return img, labels

img, *label = py_function(t, inp=[], Tout=[tf.float32] * 3)

with tf.Session() as sess:
    _img = sess.run([img])
```
但是这样执行会出现一个问题,我虽然给定的输出是3个,但是实际执行的时候函数返回值是2个:
```sh
pyfunc_10 returns 2 values, but expects to see 3 values.
```

# 问题解决

所以我需要进行返回时的解包,最终程序如下:
```sh
def t():
    img = np.zeros((240, 320, 3))
    labels = [np.array((7, 10, 3, 25)), np.array((14, 20, 3, 25))]
    return (img, *labels)

img, *label = py_function(t, inp=[], Tout=[tf.float32] * 3)

with tf.Session() as sess:
    _img, *_label = sess.run([img, *label])
```
只要注意到一点,返回`*list`是可以的,但是必须要加括号保证语法!


