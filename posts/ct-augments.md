---
title: Control Theory Augment
mathjax: true
toc: true
date: 2020-03-07 19:55:31
categories:
  - 深度学习
tags:
- Tensorflow
- 数据增强
---

`CT Augment`是论文`ReMixmatch`中提出的一种不需要通过控制方法不需要使用强化学习即可调整数据增强测量的一种方法。今天仔细学习一下。

<!--more-->


1.  初始化选择概率矩阵

首先，`CTAugment`将每个变化的每个参数范围划分为数个分组，在开始训练时将每个分组的权重设置为`1`，比如一共9种数据增强ops，数据增强分级为10级，此时权重参数`log_prob`形状为`[9,10]`。同时设置更新速率矩阵`rates`为`1`，形状为`[9,10]`。

1.  均匀随机选取数据增强方式以及数据增强分级参数

    ```python
    def _sample_ops_uniformly(self) -> [tf.Tensor, tf.Tensor]:
      """Uniformly samples sequence of augmentation ops."""
      op_indices = tf.random.uniform(
          shape=[self.num_layers], maxval=len(AUG_OPS), dtype=tf.int32)
      op_args = tf.random.uniform(shape=[self.num_layers], dtype=tf.float32)
      return op_indices, op_args
    ```
    
    均匀随机选取可以更好覆盖全部情况

2.  根据所选取的参数实施增强得到`probe_data`

3.  通过模型对`probe_data`进行分类，得到`probe_probs`

4.  使用`label`得到对应样本的正确分类`probe_probs`称为`proximity`

5.  根据公式更新`rate`矩阵
    
    此处的`op_idx, level_idx`是之前均匀随机选取的增强操作、分级参数。`decay`为衰减率默认`0.999`。
    
    ```python
    alpha = 1 - decay
    rate[op_idx, level_idx] += (proximity - rate[op_idx, level_idx]) * alpha
    ```
    
    当所得到的分类概率较高则`rate`会随之增加，反之则降低。

6.  将`rate`转换为选择概率`probs`

    ```python
    probs = tf.maximum(self.rates, self.epsilon)
    probs = probs / tf.reduce_max(probs, axis=1, keepdims=True) # 将概率锐化，类似softmax
    probs = tf.where(probs < self.confidence_threshold, tf.zeros_like(probs),
                    probs) # 如果概率小于阈值，那么设置为0
    probs = probs + self.epsilon  # 防止概率为0
    probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)  # 再次锐化
    ```

7.  将`probs`更新到`log_prob`

8.  对于训练的样本则根据`log_prob`进行数据增强参数的选取。

    ```python
    def _sample_ops(self, local_log_prob):
      """Samples sequence of augmentation ops using current probabilities."""
      # choose operations
      op_indices = tf.random.uniform(
          shape=[self.num_layers], maxval=len(AUG_OPS), dtype=tf.int32)
      # sample arguments for each selected operation
      selected_ops_log_probs = tf.gather(local_log_prob, op_indices, axis=0)
      op_args = tf.random.categorical(selected_ops_log_probs, num_samples=1)
      op_args = tf.cast(tf.squeeze(op_args, axis=1), tf.float32)
      op_args = (op_args + tf.random.uniform([self.num_layers])) / self.num_levels
      return op_indices, op_args
    ```

9.  重复以上过程。




# 总结

整个更新过程就是这样。通过选取对应的数据增强种类，得到此数据增强下的分类概率，当分类概率低时，`rate`会降低，经过锐化后此数据增强被选中的概率也会降低。其中`decay`控制了更新速率。还有`confidence_threshold`，我觉得可能要`batch`越大的时候才比较有用，如果`batch`较小很难一次性更新`rate`超过`confidence_threshold`，如果没有超过`confidence_threshold`那么此数据增强被选中的概率依旧还是比较低的。

所实话对于虽然不用强化学习的方法来更新数据增强策略了，但这两个超参数的选取还是有点头疼。并且这个控制方式缺少一定的收敛性分析。我训练半天的选取概率矩阵如下：
```python
[0.11852807, 0.13082333, 0.00013127, 0.12403152, 0.13140538, 0.00013127, 0.1205155 , 0.12174512, 0.12513067, 0.12755796],
[0.20564014, 0.00020543, 0.19176407, 0.00020543, 0.2006021 , 0.00020543, 0.20226233, 0.00020543, 0.19870412, 0.00020543],
[0.11055039, 0.11402953, 0.11110956, 0.1050452 , 0.11322882, 0.11464192, 0.11097319, 0.10542616, 0.11488046, 0.00011477],
[0.51186407, 0.48404494, 0.00051135, 0.00051135, 0.00051135, 0.00051135, 0.00051135, 0.00051135, 0.00051135, 0.00051135],
[0.14486092, 0.1384983 , 0.14066745, 0.13853313, 0.15168588, 0.1444478 , 0.14085191, 0.00015153, 0.00015153, 0.00015153],
[0.34809318, 0.00034775, 0.00034775, 0.00034775, 0.3339483 , 0.00034775, 0.00034775, 0.00034775, 0.31552422, 0.00034775],
[0.11353768, 0.11433525, 0.00011519, 0.11392737, 0.11094389, 0.10420952, 0.10411835, 0.11530466, 0.11302778, 0.11048029],
[0.0009901 , 0.0009901 , 0.0009901 , 0.0009901 , 0.0009901 , 0.0009901 , 0.0009901 , 0.0009901 , 0.99108905, 0.0009901 ],
[0.14962535, 0.15079339, 0.13698637, 0.14928676, 0.13616142, 0.13792172, 0.00015064, 0.00015064, 0.00015064, 0.13877314]
```

可视化效果如下，这个`0.99108905`我感觉很有可能是恰好上个`probe`使用了这个增强，但是一下就把概率拉到`0.99`也太夸张了把，按道理应该是越弱的增强级别概率越大才对。

![](ct-augments/probs.png)
