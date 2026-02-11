---
title: tf2.0 自定义Model高级用法
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-01-07 20:37:48
tags:
-   Tensorflow
---

最近看`insightface`深受启发，他的人脸识别在训练时可以训练`softmax`的，还可以训练`triplet loss`的，并且在验证时是对图像`pair`进行验证的。这几天弄论文顺便先给`tf`里面的模型写个骨架出来。这里的主要难点在于如何用`tf.keras`自带`Model`类实现训练和测试时不同的行为，今天尝试了一下，做个总结。


<!--more-->

## 单输出前向推导不同行为

这个比较简单，参考`dropout`层的实现方式即可。不过如果想控制训练和验证输出维度不同，这个好像暂时没有解决方案，除非使用动态图的写法。

```python
import tensorflow as tf
from tensorflow.python.keras.utils.tf_utils import smart_cond
from typing import List
import numpy as np
k = tf.keras
kl = tf.keras.layers
K = tf.keras.backend


class Model_1(k.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        """ 使用tf.cond可以控制循环，参考dropout"""
        print(training)
        if training is None:
            training = K.learning_phase()
        return smart_cond(training, lambda: tf.ones([1]), lambda: tf.zeros([1]))


def test_model_train_validation():
    """ 测试模型train与validation的不同行为，目前是成功的
    NOTE 训练和验证的输出必须是一样的 """
    train_x = tf.ones([1000, 1])
    train_y = tf.ones([1000, 1])
    md = Model_1()
    md.compile(loss=lambda y_ture, y_pred: tf.reduce_sum(y_pred))
    md.fit(train_x, train_y, batch_size=32)

    # md.predict(train_x)
    md.evaluate(train_x, train_y, batch_size=32)
```


### 多输出前向推导不同行为

如果想要一个多输出的模型，并且执行方式不同，那么这个过程还是比较蛋疼的。因为继承`Model`类的方式定义模型，我好像暂时没找到合适的指定输入输出形状的函数，这样就会导致程序不知道形状导致报错。

1.  `call`中定义多输出模型，两个输出可以维度不同，但是这样`loss`就没法复合。
2.  首先随意输入一些数据，这是为了让他自动推断输出个数与形状。
3.  手动执行`compile`，并提供`target_tensors`否则模型还是不知道输出形状。
4.  训练即可


```python

class Model_2(k.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = tf.constant(1., tf.float32)  # 1
        self.b = tf.constant(2., tf.float32)  # 2

    def call(self, inputs: List[tf.Tensor], training=None, mask=None):
        """ 测试使用多个输入时，控制不同行为"""
        a = self.a * inputs[0]
        b = smart_cond(training, lambda: self.b * inputs[1], lambda: tf.ones_like(a, tf.float32))
        return a, b

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)


def test_model_train_validation():
    """ 测试模型train与validation的不同行为下实现多输出，目前是成功的
    NOTE 训练和验证的输出维度必须一样 """
    train_x = [np.ones([1000, 1], 'float32'), np.ones([1000, 1], 'float32')]
    train_y = [np.ones([1000, 1], 'float32'), np.ones([1000, 1], 'float32')]
    md = Model_2()
    # NOTE 首先输入一次数据来build
    md.predict((train_x[0][0:1], train_x[1][0:1]))
    print(md.built)
    print(md.inputs)
    print(md.outputs)

    # NOTE 然后在compile的时候提供target_tensors
    md.compile(loss=lambda y_ture, y_pred: tf.reduce_sum(y_pred),
               target_tensors=[K.placeholder([None, 1], 1, tf.float32), K.placeholder([None, 1], 1, tf.float32)])
    print(md._is_compiled)

    md.fit(train_x, train_y, batch_size=32)
    md.evaluate(train_x, train_y, batch_size=32)
```


### 单输出前向推导、loss均不同行为

模型和之前没什么区别，在损失中需要使用`K.learning_phase`获得当前的推理状态来执行不同过程～

```python
class Model_3(k.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = kl.Dense(2)

    def call(self, inputs: List[tf.Tensor], training=None, mask=None):
        a = self.dense(inputs[0])
        b = smart_cond(training, lambda: self.dense(inputs[1]), lambda: tf.ones_like(a, tf.float32))
        return tf.stack([a, b], 1)

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)


class faceloss(k.losses.Loss):
    def __init__(self, reduction='auto', name=None):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        # y_true=[1]
        training = K.learning_phase()
        print(training)

        def false_fn():
            return tf.cast(tf.math.reduce_sum(tf.square(y_pred[:, 0] - y_pred[:, 1])) < 0.7, tf.float32)
        loss = smart_cond(training, lambda: K.sparse_categorical_crossentropy(tf.cast(y_true, tf.float32), y_pred[:, 0], True),
                          lambda: false_fn())
        return loss


def test_model_train_validation():
    """ 测试模型train与validation的不同行为，目前是成功的
    NOTE 训练和验证的输出必须是一样的 """
    train_x = [np.ones([1000, 1], 'float32'), np.ones([1000, 1], 'float32')]
    train_y = [np.ones([1000], 'int32')]

    md = Model_3()

    # NOTE 然后在compile的时候提供target_tensors
    md.compile(loss=faceloss())

    md.fit(train_x, train_y, batch_size=32)
    md.evaluate(train_x, train_y, batch_size=32)
```


## 展望

接下来应该对于要写的代码心里有数了，现在想想，最好的方式是先定义一个`Sequential`对象，然后`call`的时候反复执行这个模型即可，导出时只需导出这个`Sequential`对象。