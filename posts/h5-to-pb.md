---
title: H5模型转pb模型
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-03-17 23:10:08
tags:
- Tensorflow
- Keras
---

这个实际上是个伪需求，直接`h5`转`tflite`就好了，但是就是没办法，总有些东西不支持新的方法。下面记录一下怎样把`tf2.0`生成的`h5`模型转成`tf1.10`的`pb`模型。


<!--more-->

## 加载模型


首先用老版本的`tensorflow.keras`加载模型：
```python
import tensorflow as tf
import numpy as np
from functools import reduce, wraps
from typing import List
import os
from tensorflow.contrib.lite.python import lite
kl = tf.keras.layers
k = tf.keras
kr = tf.keras.regularizers
K = tf.keras.backend


h5_model = k.models.load_model('infer.h5')
```

当然会碰到许多不兼容的地方：

1.  `ValueError: ('Unrecognized keyword arguments:', dict_keys(['ragged']))`

这个是因为老的`k.Input`不支持`ragged`参数，修改如下：
```python
  def __init__(self,
               input_shape=None,
               batch_size=None,
               dtype=None,
               input_tensor=None,
               sparse=False,
               name=None,
               **kwargs):
    if 'batch_input_shape' in kwargs:
      batch_input_shape = kwargs.pop('batch_input_shape')
      if input_shape and batch_input_shape:
        raise ValueError('Only provide the input_shape OR '
                         'batch_input_shape argument to '
                         'InputLayer, not both at the same time.')
      batch_size = batch_input_shape[0]
      input_shape = batch_input_shape[1:]
    # NOTE 注释这里：
    # if kwargs:
    #   raise ValueError('Unrecognized keyword arguments:', kwargs.keys())
```

2.  `ValueError: Unknown initializer: GlorotUniform`

老版本的`tf.keras`没有这个初始化类，从新的`tensorflow`里面拷贝一个来就好了。

```python
class GlorotNormal(k.initializers.VarianceScaling):
  def __init__(self, seed=None):
    super(GlorotNormal, self).__init__(
        scale=1.0, mode="fan_avg", distribution="truncated_normal", seed=seed)

  def get_config(self):
    return {"seed": self.seed}
.
.
.
h5_model = k.models.load_model('infer.h5', custom_objects={'GlorotUniform': GlorotNormal})
```

3. `TypeError: ('Keyword argument not understood:', 'threshold')`

这个是因为`Relu`层的实现不一样。

先把`tf.keras.backend.relu`替换为如下：
```python
@tf_export('keras.backend.relu')
def relu(x, alpha=0., max_value=None, threshold=0):
  if alpha != 0.:
    if max_value is None and threshold == 0:
      return nn.leaky_relu(x, alpha=alpha)

    if threshold != 0:
      negative_part = nn.relu(-x + threshold)
    else:
      negative_part = nn.relu(-x)

  clip_max = max_value is not None

  if threshold != 0:
    # computes x for x > threshold else 0
    x = x * math_ops.cast(math_ops.greater(x, threshold), floatx())
  elif max_value == 6:
    # if no threshold, then can use nn.relu6 native TF op for performance
    x = nn.relu6(x)
    clip_max = False
  else:
    x = nn.relu(x)

  if clip_max:
    max_value = _to_tensor(max_value, x.dtype.base_dtype)
    zero = _to_tensor(0., x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, max_value)

  if alpha != 0.:
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x -= alpha * negative_part
  return x
```

再把`tf.keras.layers.advanced_activations`里面的`Relu`替换为如下：
```python
@tf_export('keras.layers.ReLU')
class ReLU(Layer):

  def __init__(self, max_value=None, negative_slope=0, threshold=0, **kwargs):
    super(ReLU, self).__init__(**kwargs)
    if max_value is not None and max_value < 0.:
      raise ValueError('max_value of Relu layer '
                       'cannot be negative value: ' + str(max_value))
    if negative_slope < 0.:
      raise ValueError('negative_slope of Relu layer '
                       'cannot be negative value: ' + str(negative_slope))

    self.support_masking = True
    if max_value is not None:
      max_value = K.cast_to_floatx(max_value)
    self.max_value = max_value
    self.negative_slope = K.cast_to_floatx(negative_slope)
    self.threshold = K.cast_to_floatx(threshold)

  def call(self, inputs):
    # alpha is used for leaky relu slope in activations instead of
    # negative_slope.
    return K.relu(inputs,
                  alpha=self.negative_slope,
                  max_value=self.max_value,
                  threshold=self.threshold)

  def get_config(self):
    config = {
        'max_value': self.max_value,
        'negative_slope': self.negative_slope,
        'threshold': self.threshold
    }
    base_config = super(ReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape
```


4.  `TypeError: ('Keyword argument not understood:', 'interpolation')`

这个是因为上采样层不同，替换`tensorflow/python/keras/layers/convolutional.py`里面的`UpSampling2D`：

```python
@tf_export('keras.layers.UpSampling2D')
class UpSampling2D(Layer):
  def __init__(self,
               size=(2, 2),
               data_format=None,
               interpolation='nearest',
               **kwargs):
    super(UpSampling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 2, 'size')
    if interpolation not in {'nearest', 'bilinear'}:
      raise ValueError('`interpolation` argument should be one of `"nearest"` '
                       'or `"bilinear"`.')
    self.interpolation = interpolation
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      height = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      width = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], height, width])
    else:
      height = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      width = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], height, width, input_shape[3]])

  def call(self, inputs):
    return backend.resize_images(
        inputs, self.size[0], self.size[1], self.data_format,
        interpolation=self.interpolation)

  def get_config(self):
    config = {
        'size': self.size,
        'data_format': self.data_format,
        'interpolation': self.interpolation
    }
    base_config = super(UpSampling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
```

再替换`tensorflow/python/keras/backend.py`里面的`resize_images`：

```python
@tf_export('keras.layers.UpSampling2D')
class UpSampling2D(Layer):
  def __init__(self,
               size=(2, 2),
               data_format=None,
               interpolation='nearest',
               **kwargs):
    super(UpSampling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 2, 'size')
    if interpolation not in {'nearest', 'bilinear'}:
      raise ValueError('`interpolation` argument should be one of `"nearest"` '
                       'or `"bilinear"`.')
    self.interpolation = interpolation
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      height = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      width = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], height, width])
    else:
      height = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      width = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], height, width, input_shape[3]])

  def call(self, inputs):
    return backend.resize_images(
        inputs, self.size[0], self.size[1], self.data_format,
        interpolation=self.interpolation)

  def get_config(self):
    config = {
        'size': self.size,
        'data_format': self.data_format,
        'interpolation': self.interpolation
    }
    base_config = super(UpSampling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
```

## 转换模型

加载模型好了其实就简单了：

```python
def test_convert_pb_2():

  K.clear_session()
  K.set_learning_phase(False)
  sess = K.get_session()
  init_graph = sess.graph
  input_shapes = None
  with init_graph.as_default():
    h5_model = k.models.load_model(
        'infer.h5', custom_objects={'GlorotUniform': GlorotNormal})
    input_tensors = h5_model.inputs
    output_tensors = h5_model.outputs
    lite._set_tensor_shapes(input_tensors, input_shapes)
    graph_def = lite._freeze_graph(sess, output_tensors)
    tf.train.write_graph(graph_def, 'data_2', 'model.pb', as_text=False)
    in_nodes = [inp.op.name for inp in h5_model.inputs]
    out_nodes = [out.op.name for out in h5_model.outputs]
    print(in_nodes)
    print(out_nodes)
    
test_convert_pb_2()

```

得到如下：
```sh
(vim3l) ➜  retinaface python ./test_model.py
2020-03-18 00:02:18.150198: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
['input_1']
['concatenate_3/concat', 'concatenate_4/concat', 'concatenate_5/concat']
```

然后终于可以用他的**转换工具转换了

