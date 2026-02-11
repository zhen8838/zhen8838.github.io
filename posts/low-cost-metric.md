---
title: tf.keras中优化metric计算(提取loss至metric)
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-08-28 23:02:18
tags:
-   Tensorflow
-   Keras
---

有的时候我们的`loss`函数是一个复合函数，但是在`tf.keras`中，`loss`函数只能返回一个标量，这个时候我们如果想要观察`loss`中子部分的值就只能写个`metric`去重新计算，但是这样是很浪费计算资源的，所以最好直接将`loss`中的值提取至`metric`。


<!--more-->

代码如下所示，只要我们自定义`loss`的时候给出一个变量去保存子损失的值，那么我们在`metric`可以直接对这个变量进行读值操作。

**这里还是有个小问题，就是必须要对变量的操作符进行调用才可以获得到正确的值，我已经在`tensorflow`中提交`issue`了，现在用了一个取巧的方法避开这个问题。**


```python
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kl
import tensorflow.keras.metrics as km
from tensorflow.keras.datasets import fashion_mnist

tfcfg = tf.ConfigProto()
tfcfg.gpu_options.allow_growth = True
sess = tf.Session(config=tfcfg)
k.backend.set_session(sess)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y_train = k.utils.to_categorical(y_train, 10)
y_test = k.utils.to_categorical(y_test, 10)


model = k.Sequential([
    kl.Conv2D(32, 3, 1, input_shape=[28, 28, 1]),
    kl.BatchNormalization(),
    kl.LeakyReLU(),
    kl.Conv2D(64, 3, 1),
    kl.BatchNormalization(),
    kl.LeakyReLU(),
    kl.Conv2D(128, 3, 1),
    kl.BatchNormalization(),
    kl.LeakyReLU(),
    kl.LeakyReLU(),
    kl.Flatten(),
    kl.Dense(512),
    kl.BatchNormalization(),
    kl.LeakyReLU(),
    kl.Dense(10)
])


class Metric_HIGH_COST(km.Metric):
    def __init__(self, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.ce = self.add_weight('ce', initializer=tf.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.ce.assign(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)))

    def result(self):
        return self.ce


class Metric_LOW_COST(km.Metric):
    def __init__(self, cross_entropy: tf.Variable, name='CE', dtype=None):
        """ yolo landmark error metric

        Parameters
        ----------
        MeanMetricWrapper : [type]

        landmark_error : ResourceVariable
            a variable from yoloalign loss
        name : str, optional
            by default 'LE'
        dtype : [type], optional
            by default None
        """
        super().__init__(name=name)
        self.ce = cross_entropy

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.ce

    def result(self):
        return self.ce.read_value()


class Myloss(k.losses.Loss):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.ce = tf.get_variable('ce', (), tf.float32, tf.zeros_initializer)  # type:tf.RefVariable

    def call(self, y_true, y_pred):
        ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

        # ! method 1 got zero output :
        # self.ce.assign(ce_loss)
        # return ce_loss

        # ! method 2 get correct output :
        return ce_loss + 0 * self.ce.assign(ce_loss)


myloss = Myloss()
high_cost_metric = Metric_HIGH_COST('high_cost_ce')
low_cost_metric = Metric_LOW_COST(myloss.ce, 'low_cost_ce')

sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

model.compile(k.optimizers.Adam(), [myloss], [high_cost_metric, low_cost_metric])

model.fit(x_train, y_train, 100, 10)
```

