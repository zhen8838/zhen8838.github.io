---
title: 条件VAE
mathjax: true
toc: true
categories:
-   深度学习
date: 2019-08-18 16:06:01
tags:
-   Tensorflow
-   VAE
---

这几天时间自己把`TensorFlow Probability`里面的几个例子过了一遍，希望以后可以做出一些深度学习与概率论结合的成果。


今天我试着用`TensorFlow Probability`把条件VAE实现一下。这个条件VAE通过控制传统`VAE`中的正态分布的均值来达到分类生成的效果，这样每个类别都有一个专属均值，可以通过这个专属均值来生成与此类相似的结果。

<!--more-->

# 条件VAE

下面直接上代码了，利用好`TensorFlow Probability`这个神器写起来是十分方便的，但是原理自己还是必须要懂才可以，如果不理解建议看看苏剑林的[文章](https://kexue.fm/archives/5253)。


```python
# %%
import numpy as np
import tensorflow.python as tf
import tensorflow.python.keras as tfk
import tensorflow.python.keras.layers as tfkl
import tensorflow_probability.python as tfp
import tensorflow_probability.python.layers as tfpl
import tensorflow_probability.python.distributions as tfd
from toolz import compose, pipe
from scipy.stats import norm
import matplotlib.pyplot as plt
tf.enable_eager_execution()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tfk.backend.set_session(tf.Session(config=config))


# %%
(x_train, y_train), (x_test, y_test) = tfk.datasets.mnist.load_data()


def _preprocess(x, y):
    x = tf.reshape(x, (28, 28, 1))
    x = tf.cast(x, tf.float32) / 255.  # Scale to unit interval.
    y = tf.one_hot(y, 10)
    return (x, y), ()


train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(int(10000))
                 .map(_preprocess)
                 .batch(256)
                 .prefetch(tf.data.experimental.AUTOTUNE))

eval_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                .map(_preprocess)
                .batch(256)
                .prefetch(tf.data.experimental.AUTOTUNE))

# %%

input_shape = [28, 28, 1]
encoded_size = 2
base_depth = 16
num_class = 10


# %% 构建先验
# 把正态分布的均值也修改为可训练的,利用y去训练分布的均值
y_inputs = tfk.Input(shape=(num_class), name='y_inputs')

cluster_mean = tfk.Sequential([
    tfkl.InputLayer(num_class),
    tfkl.Dense(encoded_size * 2),
    tfkl.Dense(encoded_size)])

mean = cluster_mean(y_inputs)

prior = tfd.Independent(tfd.Normal(loc=mean, scale=1),
                        reinterpreted_batch_ndims=1)


# %% 构建编码器
x_inputs = tfk.Input(shape=input_shape, name='x_inputs')
# encoder
encoder = tfk.Sequential(
    [tfkl.Conv2D(base_depth, 5, strides=1,
                 padding='same', activation=tf.nn.leaky_relu, input_shape=input_shape),
     tfkl.Conv2D(base_depth, 5, strides=2,
                 padding='same', activation=tf.nn.leaky_relu),
     tfkl.Conv2D(2 * base_depth, 5, strides=1,
                 padding='same', activation=tf.nn.leaky_relu),
     tfkl.Conv2D(2 * base_depth, 5, strides=2,
                 padding='same', activation=tf.nn.leaky_relu),
     tfkl.Conv2D(4 * encoded_size, 7, strides=1,
                 padding='valid', activation=tf.nn.leaky_relu),
     tfkl.Flatten(),
     # 提供给下面一层的参数个数需要用params_size来计算。
     tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size), activation=None),
     # 一个多元正态分布层,这个MultivariateNormalTriL本身是带相关性的多元正态分布，利用KL正则的方式将其分布趋向独立标准正态分布
     tfpl.MultivariateNormalTriL(
        encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
     ])

encoder_outputs = encoder(x_inputs)

# %% 构建解码器

decoder_input = tfk.Input((encoded_size), name='decoder_input')

decoder = tfk.Sequential([
    tfkl.Reshape((1, 1, encoded_size), input_shape=[encoded_size]),
    tfkl.Conv2DTranspose(2 * base_depth, 7, strides=1,
                         padding='valid', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(2 * base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(2 * base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(filters=1, kernel_size=5, strides=1,
                padding='same', activation=tfk.activations.sigmoid),
])


decoder_outputs = decoder(encoder_outputs)
restruct_loss = tfk.backend.sum(tfk.backend.binary_crossentropy(x_inputs, decoder_outputs))


# %% 构建模型

vae = tfk.Model([x_inputs, y_inputs], [decoder_outputs, mean])
vae.add_loss(restruct_loss)
vae.compile('adam')
vae.summary()

# %% 训练

vae.fit(train_dataset, epochs=15, validation_data=eval_dataset, verbose=0)

# %% 输出每个类的均值向量

class_mean = cluster_mean(np.eye(num_class))


# %% 利用每个类的均值去采样分布

# 观察能否通过控制隐变量的均值来输出特定类别的数字
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

digit = 8  # 指定输出数字

# 用正态分布的分位数来构建隐变量对
grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) + class_mean[digit][1]
grid_y = norm.ppf(np.linspace(0.05, 0.95, n)) + class_mean[digit][0]

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
```


## 结果

1.  生成类别8

![](cvae/8.png)

2.  生成类别9

![](cvae/9.png)
