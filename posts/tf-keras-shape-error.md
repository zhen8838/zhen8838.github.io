---
title: tf.dataset无法推断shape导致错误
categories:
  - 深度学习
date: 2019-06-10 21:22:43
tags:
-   Tensorflow
-   踩坑经验
-   Keras
---

使用`tensorflow.keras`的时候，`tf.dataset`在执行`model.fit`的时候报错：

    ValueError: Cannot take the length of shape with unknown rank.

这里大概率是因为`tf.dataset`中使用了`tf.py_function`导致无法自动推导出张
良的形状，所以需要自己手动设置形状。

<!--more-->

# 解决方案

这里一定要使用`tensorflow` 1.x版本，2.0中我也没找到解决方案😓,使用`tf.contrib.data.assert_element_shape`
函数直接指定形状即可。

```python
import tensorflow as tf
from tensorflow.python import keras

yolo_model = keras_yolo_mobilev2((240, 320, 3), 3, 20, 1., True)

shapes = (yolo_model.input.shape, tuple(out.shape for out in yolo_model.output))
h.train_dataset = h.train_dataset.apply(tf.contrib.data.assert_element_shape(shapes))

yolo_model.fit(h.train_dataset, epochs=max_nrof_epochs, 
                steps_per_epoch=h.train_epoch_step,callbacks=[tbcall])
```
