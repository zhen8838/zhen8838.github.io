---
title: tf.keras多输出模型自定义loss
categories:
-   深度学习
date: 2019-06-09 16:43:46
tags:
-   Tensorflow
-   Keras
---

自从看了苏剑林的博客之后,我对`keras`是越来越喜欢了,但是我更喜欢在`tensorflow`中使用`keras`,今天就来看看如何在`tf.keras`中自定义多输出模型的`loss`,并且搭配高效的`tf.dataset`.

**NOTE:** `tensorflow==2.0.0b0`

<!--more-->

# 解决方案

## 1.使用自定义loss函数的方式

`keras`中使用自定义`loss`函数后,他是自动将每一个输出与标签进行误差计算,这样的话要求你的`loss`函数必须适用到每一个输出,当然也可以根据输出尺寸什么的做一些动态调整,总的来说灵活度还差点.

这样的好处是变量用的少,并且`tf.dataset`的操作比较方便,不需要写太多.

```python
import tensorflow.python as tf
from tensorflow.python import keras
import numpy as np
""" 这个方式比较优雅,但是loss函数必须对所有输出适用 """
keras.backend.clear_session()
x = keras.Input(shape=(10))
x_1 = keras.layers.Dense(10)(x)
x_2 = keras.layers.Dense(10)(x)
model = keras.Model(inputs=x, outputs=[x_1, x_2])
model.summary()

def l(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return y_true - y_pred

train_x = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 10)).repeat().batch(32)
train_y = tf.data.Dataset.from_tensor_slices((np.random.rand(100, 10), np.random.rand(100, 10))).repeat().batch(32)

train_set = tf.data.Dataset.zip((train_x, train_y))

model.compile('adam', l)

model.fit(train_set, steps_per_epoch=30)
```

## 2.使用add_loss的方式

这个方式更加灵活,但是必须要把输入输出都用到`loss`里面,这样一开始的灵活可能会造成后面的阻碍,并且对`tf.dataset`的处理也造成了一些困扰.

```python
import tensorflow.python as tf
from tensorflow.python import keras
import numpy as np
keras.backend.clear_session()

x = keras.Input(shape=(10))
label_1 = keras.Input(shape=(10))
label_2 = keras.Input(shape=(10))
pred_1 = keras.layers.Dense(10)(x)
pred_2 = keras.layers.Dense(10)(x)

model = keras.Model(inputs=[x, label_1, label_2], outputs=[pred_1, pred_2])
model.summary()

train_set = tf.data.Dataset.from_tensor_slices({'input_1': np.random.rand(100, 10), 'input_2': np.random.rand(100, 10), 'input_3': np.random.rand(100, 10)}).repeat().batch(32)
train_set = train_set.map(lambda x: {'input_1': x['input_1'] + .1, 'input_2': x['input_2'] - .2, 'input_3': x['input_3'] - .5})

def losses(labels: list, preds: list):
    l = 0
    for i in range(len(labels)):
        # 这里我可以给不同的label不同的loss操作
        l += tf.reduce_sum(((labels[i] - preds[i])**2) * (i + 1))
    return l

model.add_loss(losses([label_1, label_2], [pred_1, pred_2]))
model.compile('adam')

model.fit(train_set, steps_per_epoch=30)
```