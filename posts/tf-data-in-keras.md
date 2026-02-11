---
title: tf.Keras完美使用tf.data API
categories:
  - 深度学习
date: 2019-05-21 16:03:46
tags:
-   Tensorflow
-   Keras
---

最近看了苏剑林的几篇[博客](https://kexue.fm/search/%E8%AE%A9Keras%E6%9B%B4%E9%85%B7%E4%B8%80%E4%BA%9B/)，
我忽然对`keras`不是那么抵触了,才发现之前认为`Keras`使用不灵活完全是因为的认识不够深入。所以我准备使用`Tensorflow 2.0`中的`tf.Keras`来
构建`Yolo v3`，在`tensorflow`中我们可以更加灵活的优化我们的数据输入管道，这次介绍一下多输入的`model`如何结合`tf.data`，基础的使用方式在[这里](https://www.tensorflow.org/alpha/tutorials/load_data/csv)学习。

<!--more-->


# 起因

大家都知道，`YOLO v3`是一个多输出的模型，在构建模型的时候，`label`需要多个，所以模型输入也是多个。如果输入是`numpy`数组，可以保存不同维度的数组，但是`tf.Tensor`只能保存相同尺寸的数组。

然后在`model.fit`中使用`tf.data`,一般情况下是保证返回值对应`x、y`,但是现在`yolo`需要输入`[x1,x2,x3,x4],y`,这需要`tf.data`返回一个包含元组的列表。

# 解决方案

首先我尝试在`map`中直接返回嵌套的列表，但是因为我使用的是`py_function`，返回值不能嵌套(暂时没有尝试纯tensorflow的函数能否返回嵌套列表)。还好找了一个好用的方法：`tf.data.dataset.zip`，可以将两个`dataset`对象制作成嵌套列表，所以只需要小改动程序如下，把样本与标签分开制作，然后再合并即可！：

```python
def parser(lines):
    image_data = []
    box_data = []
    for line in lines:
        image, box = get_random_data(line.numpy().decode(), input_shape, random=True)
        image_data.append(image)
        box_data.append(box)

    image_data = np.array(image_data)
    box_data = np.array(box_data)

    y_true = [tf.convert_to_tensor(y, tf.float32) for y in preprocess_true_boxes(box_data, input_shape, anchors, num_classes)]
    image_data = tf.convert_to_tensor(image_data, tf.float32)
    return (image_data, *y_true)

x_set = (tf.data.Dataset.from_tensor_slices(annotation_lines).
         apply(tf.data.experimental.shuffle_and_repeat(batch_size * 300, seed=66)).
         batch(batch_size, drop_remainder=True).
         map(lambda lines: py_function(parser, [lines], [tf.float32] * (1 + len(anchors) // 3))))
y_set = tf.data.Dataset.from_tensors(tf.zeros(batch_size, tf.float32)).repeat()
dataset = tf.data.Dataset.zip((x_set, y_set))

sample = next(iter(dataset))
。
。
。
。
train_set = create_dataset(lines[:num_train], batch_size, input_shape, anchors, num_classes)

model.fit(train_set,
          epochs=20,
          steps_per_epoch=max(1, num_train // batch_size),
          callbacks=[logging, checkpoint])
```
