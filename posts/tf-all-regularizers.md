---
title: tf2.0 全局添加regularizers
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-02-12 15:11:47
tags:
- Tensorflow
---

如何给`tensorflow`中预训练好的模型添加正则化器？

<!--more-->

由于`tensorflow`基于图的定义方式，在定义好模型后，再添加正则化器是无效的，必须要重新建立图才可以。使用`tf.keras.models.model_from_json(model.to_json())`是一种方法，不过如果是加载的预训练模型，我们还需要重新加载权重才可以。

```python
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
inp = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inp)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(10)(x)
model = tf.keras.Model(inp, x)

for layer in model.layers:
    for attr in ['kernel_regularizer']:
        if hasattr(layer, attr):
            setattr(layer, attr, tf.keras.regularizers.l2(0.004))

train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))
model = tf.keras.models.model_from_json(model.to_json())

train_out = model(train_data, training=True)
print(model.losses)
```