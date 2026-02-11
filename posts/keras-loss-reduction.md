---
title: tf.keras损失函数聚合测试
mathjax: true
toc: true
date: 2019-09-22 13:07:00
categories:
  - 深度学习
tags:
-   Tensorflow
-   Keras
---

使用`tf.keras`自定义损失函数的时候,他的`reduction`的文档解释太少,所以写个代码测试一下预期的行为.

<!--more-->


## 代码

```python
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.layers as kl
from tensorflow.python.keras.utils.losses_utils import ReductionV2

x = np.reshape(np.arange(10000), (1000, 10))
y = np.tile(np.array([[1, 2, 3]], np.float32), [1000, 1])

model = k.Sequential([kl.Dense(3, input_shape=[10])])  # type:k.Model

class test_Loss(k.losses.Loss):
    def __init__(self, reduction=ReductionV2.SUM_OVER_BATCH_SIZE, name=None):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return y_true + 0 * y_pred

print('AUTO : ')
model.compile(k.optimizers.Adam(), [test_Loss(ReductionV2.AUTO)])
model.fit(x, y, 100)
print('NONE : ')
model.compile(k.optimizers.Adam(), [test_Loss(ReductionV2.NONE)])
model.fit(x, y, 100)
print('SUM : ')
model.compile(k.optimizers.Adam(), [test_Loss(ReductionV2.SUM)])
model.fit(x, y, 100)
print('SUM_OVER_BATCH_SIZE : ')
model.compile(k.optimizers.Adam(), [test_Loss(ReductionV2.SUM_OVER_BATCH_SIZE)])
model.fit(x, y, 100)
```

## 结果

```sh
AUTO : 
1000/1000 [==============================] - 1s 757us/sample - loss: 2.0000
NONE : 
1000/1000 [==============================] - 0s 79us/sample - loss: 2.0000
SUM : 
1000/1000 [==============================] - 0s 88us/sample - loss: 600.0000
SUM_OVER_BATCH_SIZE : 
1000/1000 [==============================] - 0s 105us/sample - loss: 2.0000
```

## 解析

主要就是`SUM`与`SUM_OVER_BATCH_SIZE`,这里的`SUM_OVER_BATCH_SIZE`其实是先求和再除以的`batch size`是`total loss`中的元素个数,并不是训练的`batch size`.