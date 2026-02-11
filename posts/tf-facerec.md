---
title: tensorflow人脸识别
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-01-14 13:07:21
tags:
-   Tensorflow
-   人脸识别
---

在此记录一下参考`insightface`用`tensorflow`实现人脸识别的过程。

<!--more-->

# 难点1: tf.keras不支持动态切换输出

&emsp;&emsp;训练的时候需要使用`softmax`进行训练，但是验证的时候需要输出`embedding`层的结果进行距离比较判断是否为同一个人。使用`tf.keras`的方式无法做到方便的动态切换，因此我写了上面一篇博客“[tf2.0 自定义Model高级用法](https://zhen8838.github.io/2020/01/08/tf-custom-model/)”。但是发现这样还是没法让训练和测试时输出维度不同，后来在群里问了苏神，可以直接分成3个模型`infer_model, val_model, train_model`，并删除`train_model.fit()`时期的验证，自己重新定义一个`callback`对`val_model`进行验证，最后导出时使用`infer_model`。

```python
class FacerecValidation(k.callbacks.Callback):
    def __init__(self, validation_model: k.Model, validation_data: tf.data.Dataset, validation_steps: int,
                 distance_fn: str, threshold: float):
        self.val_model = validation_model
        self.val_iter: Iterable[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]] = iter(validation_data)
        self.val_step = validation_steps
        self.distance_fn: l2distance = distance_register[distance_fn]
        self.threshold = threshold

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        acc: List[tf.Tensor] = []
        for i in range(validation_steps):
            x, actual_issame = next(self.val_iter)  # actual_issame:tf.Bool
            y_pred: Tuple[tf.Tensor] = self.val_model.predict(x)
            dist = self.distance_fn(*y_pred)  # [batch]
            pred_issame = tf.less(dist, self.threshold)

            tp = tf.reduce_sum(tf.logical_and(pred_issame, actual_issame))
            fp = tf.reduce_sum(tf.logical_and(pred_issame, tf.logical_not(actual_issame)))
            tn = tf.reduce_sum(tf.logical_and(tf.logical_not(pred_issame), tf.logical_not(actual_issame)))
            fn = tf.reduce_sum(tf.logical_and(tf.logical_not(pred_issame), actual_issame))

            tpr = tf.math.divide_no_nan(tf.cast(tp, tf.float32), tf.cast(tp + fn, tf.float32))
            fpr = tf.math.divide_no_nan(tf.cast(fp, tf.float32), tf.cast(fp + tn, tf.float32))
            acc.append[tf.cast(tp + tn, tf.float32) / tf.cast(dist.size, tf.float32)]
        acc = tf.reduce_mean(acc)
        logs['val_acc'] = acc.numpy()
        
        return super().on_epoch_end(epoch, logs=logs)

```


# 难点2: tf.data难以输出triplet对数据

&emsp;&emsp;本来想用传统的索引+原始图片的方式输入数据，但是索引矩阵长就`58`万了，我的电脑构建好数据管道之后直接爆炸。。。然后我就思考怎么用`tfrecord`的方式构建`triplet对`，对于训练`softmax`来说比较简单，既可以把所有图片都做成一个`tfrecord`，还可以把每个`id`对应的图片做成一个`tfrecord`，里面只要包含图像和标签数据即可。

&emsp;&emsp;但是所有图片做成一个`tfrecord`肯定是不行的，因为咱们没办法在`tfrecord`里面根据索引进行查找图像。这里我真的很想换到`mxnet`，他的`iamgerecorditer`是高度压缩且带索引的格式，`tf`为何就是不能把`tfrecord`里面的索引暴露出来，哪怕一个索引对应一块数据也可以啊。。但现在只能退而求其次，把每个`id`对应一个`tfrecord`，然后使用`tf.data`里面的函数进行构建。

&emsp;&emsp;一番尝试之后，构建代码如下，首先`h.train_list`包含了所有`tfrecord`的路径，然后利用`interleave`将所有的路径转换为`dataset`对象，这里需要设置`block_length=2`，这样就保证每个采样周期为2，接着再使用`batch(4)`，这样我们可以从两个不同类别中分别采样两个样本，此时得到了4张图像，取前三张即可满足要求。

```python
h = FcaeRecHelper('data/ms1m_img_ann.npy', [112, 112], 128, use_softmax=False)
len(h.train_list)
img_shape = list(h.in_hw) + [3]

is_augment = True
is_normlize = False

def parser(stream: bytes):
    examples: dict = tf.io.parse_single_example(
        stream,
        {'img': tf.io.FixedLenFeature([], tf.string),
         'label': tf.io.FixedLenFeature([], tf.int64)})
    return tf.image.decode_jpeg(examples['img'], 3), examples['label']

def pair_parser(raw_imgs, labels):
    # imgs do same augment ~
    if is_augment:
        raw_imgs, _ = h.augment_img(raw_imgs, None)
    # normlize image
    if is_normlize:
        imgs: tf.Tensor = h.normlize_img(raw_imgs)
    else:
        imgs = tf.cast(raw_imgs, tf.float32)

    imgs.set_shape([4] + img_shape)
    labels.set_shape([4, ])
    # Note y_true shape will be [batch,3]
    return (imgs[0], imgs[1], imgs[2]), (labels[:3])

batch_size = 1
ds = (tf.data.Dataset.from_tensor_slices(h.train_list)
      .interleave(lambda x: tf.data.TFRecordDataset(x)
                  .shuffle(100)
                  .repeat(), cycle_length=-1,
                  block_length=2,
                  num_parallel_calls=-1)
      .map(parser, -1)
      .batch(4, True)
      .map(pair_parser, -1)
      .batch(batch_size, True))

iters = iter(ds)
for i in range(20):
    imgs, labels = next(iters)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(imgs[0].numpy().astype('uint8')[0])
    axs[1].imshow(imgs[1].numpy().astype('uint8')[0])
    axs[2].imshow(imgs[2].numpy().astype('uint8')[0])
    plt.show()
```



