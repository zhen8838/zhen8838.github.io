---
title:  tf.keras自定义loss报错shape mismatch
categories:
  - 深度学习
date: 2019-06-11 15:13:44
tags:
-   Tensorflow
-   踩坑经验
---

在上次的[文章](https://zhen8838.github.io/2019/06/10/tf-keras-mult-out/)中我写了如何自定义`loss`，但是我真正想要的使用的场景比那些还要复杂一些。

<!--more-->

# 问题出现

是想在自定义`loss`函数中对`y_pred`进行`reshape`然后进行`sigmoid_cross_entropy_with_logits`，但是`keras`他是将`loss`构建成一个`graph`，并且`loss`中的`y_true`并不是占位符，他的`shape`是根据模型最终的输出维度来确定的**虚占位符**，如果模型最后输出的维度和真正的`y_true`维度不匹配，那么是肯定报错的。

看下面这个例子：

```python
import tensorflow as tf
from tensorflow.python import keras
import numpy as np
keras.backend.clear_session()
x = keras.Input(shape=(10))
x_1 = keras.layers.Dense(35)(x)
x_2 = keras.layers.Dense(70)(x)
model = keras.Model(inputs=x, outputs=[x_1, x_2])
model.summary()

def l_1(true, pred):
    pred = tf.reshape(pred, (-1, 5, 7))
    print(true.shape, pred.shape)
    # NOTE reshape之后 shape是匹配的，但是检查维度时候会报错
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=pred))

def l_2(true, pred):
    pred = tf.reshape(pred, (-1, 10, 7))
    print(true.shape, pred.shape)
    # NOTE reshape之后 shape是匹配的，但是检查维度时候会报错
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=pred))

train_set = tf.data.Dataset.from_tensor_slices((np.random.rand(100, 10), np.random.rand(100, 5, 7),
                                                np.random.rand(100, 10, 7))).repeat()  # type: tf.data.Dataset
train_set = train_set.map(lambda x, y, z: (x, (y, z))).batch(32)

model.compile('adam', loss=[l_1, l_2])
model.fit(train_set, steps_per_epoch=30)  # NOTE 不可训练
```

输出：

```sh
(?, ?) (?, 5, 7)
ValueError: logits and labels must have the same shape ((?, 5, 7) vs (?, ?))
```

可以看到我打印出`y_true`的`shape`就是和网络输出尺寸相同的，但实际上`y_true`输入是正确的，这个实在是让人蛋疼。

# 错误解决

因为`y_true`的`shape`就是和网络输出尺寸相同，所以就从网络上面下手，构建一个`model_warrper`用于训练。等待训练完成之后直接保存`model`即可。

```python
import tensorflow as tf
from tensorflow.python import keras
import numpy as np
keras.backend.clear_session()
x = keras.Input(shape=(10))
x_1 = keras.layers.Dense(35)(x)
x_2 = keras.layers.Dense(70)(x)
model = keras.Model(inputs=x, outputs=[x_1, x_2])
model.summary()


def l_1(true, pred):
    pred = tf.reshape(pred, (-1, 5, 7))
    print(true.shape, pred.shape)
    # NOTE reshape之后 shape是匹配的，但是检查维度时候会报错
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=pred))


def l_2(true, pred):
    pred = tf.reshape(pred, (-1, 10, 7))
    print(true.shape, pred.shape)
    # NOTE reshape之后 shape是匹配的，但是检查维度时候会报错
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=pred))


train_set = tf.data.Dataset.from_tensor_slices((np.random.rand(100, 10), np.random.rand(100, 5, 7),
                                                np.random.rand(100, 10, 7))).repeat()  # type: tf.data.Dataset
train_set = train_set.map(lambda x, y, z: (x, (y, z))).batch(32)

# model.compile('adam', loss=[l_1, l_2])
# model.fit(train_set, steps_per_epoch=30)  # NOTE 不可训练

x_1 = keras.layers.Reshape((5, 7))(x_1)
x_2 = keras.layers.Reshape((10, 7))(x_2)
model_warpper = keras.Model(inputs=x, outputs=[x_1, x_2])
model_warpper.summary()
model_warpper.compile('adam', loss=[l_1, l_2])
model_warpper.fit(train_set, steps_per_epoch=30)  # NOTE 可训练
```
