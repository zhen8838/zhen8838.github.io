---
title: tf.keras.Model.outputs隐藏问题
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-05-15 09:51:39
tags:
- Tensorflow
- Keras
---

今天发现`tf.keras.Model.outputs`的隐藏问题(feature),我居然之前都没有发现233

<!--more-->

# 描述

其实就是我这次的模型输出的是`List[Tuple]`的方式,然后我以为`keras`的模型输出还是`List`的形式,算`loss`的时候就一直出错.

```python
train_model.outputs
[<tf.Tensor 'MConv_Stage1_L1_5_bn/Identity:0' shape=(None, 80, 120, 38) dtype=float32>,
 <tf.Tensor 'MConv_Stage1_L2_5_bn/Identity:0' shape=(None, 80, 120, 19) dtype=float32>,
 <tf.Tensor 'MConv_Stage2_L1_5_bn/Identity:0' shape=(None, 80, 120, 38) dtype=float32>,
 <tf.Tensor 'MConv_Stage2_L2_5_bn/Identity:0' shape=(None, 80, 120, 19) dtype=float32>,
 <tf.Tensor 'MConv_Stage3_L1_5_bn/Identity:0' shape=(None, 80, 120, 38) dtype=float32>,
 <tf.Tensor 'MConv_Stage3_L2_5_bn/Identity:0' shape=(None, 80, 120, 19) dtype=float32>,
 <tf.Tensor 'MConv_Stage4_L1_5_bn/Identity:0' shape=(None, 80, 120, 38) dtype=float32>,
 <tf.Tensor 'MConv_Stage4_L2_5_bn/Identity:0' shape=(None, 80, 120, 19) dtype=float32>,
 <tf.Tensor 'MConv_Stage5_L1_5_bn/Identity:0' shape=(None, 80, 120, 38) dtype=float32>,
 <tf.Tensor 'MConv_Stage5_L2_5_bn/Identity:0' shape=(None, 80, 120, 19) dtype=float32>]
```

然后我发现实际上模型的输出就是按原本的形式:

```python
train_model.output_shape
[((None, 80, 120, 38), (None, 80, 120, 19)),
 ((None, 80, 120, 38), (None, 80, 120, 19)),
 ((None, 80, 120, 38), (None, 80, 120, 19)),
 ((None, 80, 120, 38), (None, 80, 120, 19)),
 ((None, 80, 120, 38), (None, 80, 120, 19))]
```

结论就是`tf.keras`比我想的完善多了233