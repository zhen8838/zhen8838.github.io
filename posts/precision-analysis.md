---
title: yolo中precision降低的原因分析
categories:
  - 深度学习
date: 2019-03-03 10:26:18
tags:
-   踩坑经验
-   Tensorflow
-   Yolo
---

最近在训练`yolo`模型一类检测模型,我在训练过程中发现`precision`会降低,思考之后对其做出一些分析.

<!--more-->

# 起因

我在训练`yolo`的时候,首先是进行分类检测,即只训练`class_loss`,`obj_loss`,`noobj_loss`,这个时候训练`precision`会达到`60%`.

接下来进行第二次迭代,这次同时训练`class_loss`,`obj_loss`,`noobj_loss`,`xy_loss`,`wh_loss`,这个时候`precision`会降低到`40%`.

如下图所示:
![](precision-analysis/1.jpg)

# 原因分析

经过思考,我找到问题所在:
```sh
""" calc the noobj mask ~ """
if train_classifier == 'True':
    noobj_mask = tf.logical_not(obj_mask)
else:
    noobj_mask = calc_noobj_mask(true_xy, true_wh, pred_xy, pred_wh, obj_mask, iou_thresh=iou_thresh, helper=helper)
```

这段代码就是当我开始训练`xy_loss`,`wh_loss`时,那么`noobj_mask`就需要通过预测出来的`box`与`ground truth box`计算`iou`来筛选.

当我可以拟合`box`之后,那么也就是说如果其他的`cell`预测出来的`box`与`ground truth box`的`iou`大于`iou_threshold`之后,那么这个`cell`我就不会将他设置为`noobj cell`,同时也不会对这个`cell`进行惩罚措施.
![](precision-analysis/2.jpeg)


再看`precision`和`recall`的计算方式:
$$ 
\begin{aligned}
precision&=\frac{true\ positive}{true\ positive+false\ positive} \\
recall&=\frac{true\ positive}{true\ positive+false\ negative}
\end{aligned}
$$

因为我没有对预测`box`与`true box`的`iou`大于`iou_threshold`进行惩罚,所以我们的$false\ positive$会增加,$false\ negative$会减少.最终导致`precision`减少,`recall`增加.