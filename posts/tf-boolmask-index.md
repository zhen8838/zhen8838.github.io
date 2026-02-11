---
title: tf2.0得到子boolmask在boolmask中的索引
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-12-13 16:39:59
tags:
- Tensorflow
---

在`yolo`中计算了单层的`anchor`与全局的`gt`间的`iou score`，但是我需要在其中过滤出单层的`anchor`对应单层的`gt`的`iou score`。目前有单一层的`gt`的`loc_mask`，以及全局的`gt`的`glob_mask`，其中`loc_mask`中有效区域是`glob_mask`的子集，因此需要找到`loc_mask`在`glob_mask`的对应索引。

<!--more-->


# 问题描述

前面讲的可能比较抽象，所以给出一个例子：
```python
iou = tf.random.normal((4, 4, 9))
loc_mask = tf.constant([[False, False, False, False, ],
                        [False, True, False, False, ],
                        [False, False, False, False, ],
                        [False, False, True, False, ]])
glob_mask = tf.constant([[False, False, True, False, ],
                         [False, True, False, True, ],
                         [True, False, False, False, ],
                         [False, True, True, False, ]])
loc_iou = tf.boolean_mask(iou, loc_mask)  # (2, 9)
glob_iou = tf.boolean_mask(iou, glob_mask)  # (6, 9)
```

实际上就是要找到`loc_iou`中元素对应`glob_iou`中的位置。如果没有`loc_mask`和`glob_mask`实际上很难找到。

## 问题解决

我想了半天，终于发现一个简单的方式，先利用`where`和`boolmask`找到有效位置，再使用`gather_nd`就完事了：

```python
iou = tf.random.normal((4, 4, 9))
loc_mask = tf.constant([[False, False, False, False, ],
                        [False, True, False, False, ],
                        [False, False, False, False, ],
                        [False, False, True, False, ]])
glob_mask = tf.constant([[False, False, True, False, ],
                         [False, True, False, True, ],
                         [True, False, False, False, ],
                         [False, True, True, False, ]])
loc_iou = tf.boolean_mask(iou, loc_mask)  # (2, 9)
glob_iou = tf.boolean_mask(iou, glob_mask)  # (6, 9)

idx = tf.where(tf.boolean_mask(loc_mask, glob_mask)) # (2, 1) [[1],[5]]
mask_iou = tf.gather_nd(tf.boolean_mask(iou, glob_mask), idx, 0) # (2, 9)
```

