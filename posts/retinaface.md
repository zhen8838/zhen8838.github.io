---
title: retinaface总结
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-12-19 14:42:44
tags:
-   Retinaface
-   Tensorflow
-   目标检测
---

&emsp;&emsp;本次主要总结一下[retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace)和[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)。

&emsp;&emsp;实际上`retinaface`和`Ultra-Light-Fast-Generic-Face-Detector-1MB`的思路都是基于`SSD`的，本来我做`yolo`之后准备学习一下`SSD`的，做完这两个模型也算是学习到了。由于我目前不开源基于`tensorflow`的训练代码，下面的代码大家仅供参考～

<!--more-->

# 网络设计

&emsp;&emsp;骨干网络实际上随便来，主要就是`SSD`预测层和`YOLO`不太一样。

-   YOLO的预测层

    只使用一个`Conv2D 1×1`得到`anchor_num * (class_num + 5)`的输出。
    
-   SSD的预测层

    如果只有`bbox`输出和`class`输出，那么使用两个`Conv2D 1×1`分别得到`anchor_num * 4`和`anchor_num * class_num`。不过这样卷积的参数实际上相同的，谁好谁坏得实验下才能知道了。



&emsp;&emsp;这里的标签与`YOLO`中不太一样，`YOLO`的标签制作可以看这篇[文章](https://zhen8838.github.io/2019/07/10/yolo-error/)。前面的图像增强啥我就不讲了，不过我之前在`YOLO`里面用了多尺度训练，但是实际上用动态图像裁剪缩放也可以得到相同的效果，所以这里我就没有再用多尺度训练了。

#  `anchor`生成

&emsp;&emsp;这个`anchor`生成和`YOLO`中的`anchor`生成方式类似，只不过这里生成的`anchor`还需要加上中心点的位置，`grid`的位置加0.5然后乘上比例：

```python
cx = (j + 0.5) * steps[k] / image_size[1]
cy = (i + 0.5) * steps[k] / image_size[0]
```

`anchor`的宽度则是当前`feature map`指定的`anchor`宽度除以总宽度

```python
s_kx = min_size[1] / in_hw[1]
s_ky = min_size[0] / in_hw[0]
```

然后把几个`feature map`上生成的`anchor`全部连接起来：

```python
output = np.concatenate([
    np.reshape(anchors[0], (-1, 4)),
    np.reshape(anchors[1], (-1, 4)),
    np.reshape(anchors[2], (-1, 4))], 0)
```

我觉得这样比`YOLO`中的方式好，`YOLO`中计算`loss`的时候需要计算全体的`gt`，但是算的时候是分开算的，比较蛋疼。

# 标签制作

1.  首先计算`gt`与`anchor`的`iou`
    
    ```python
    overlaps = tf_bbox_iou(bbox, self.corner_anchors)
    ```
    
2.  找到大于阈值的的`anchor`位置，这个阈值可以设置的小一点，不过太小也不好，那样`label`中的匹配`anchor`数量就会太多
    
    ```python
    best_prior_overlap = tf.reduce_max(overlaps, 1)
    best_prior_idx = tf.argmax(overlaps, 1, tf.int32)
    best_prior_idx_filter = tf.boolean_mask(best_prior_idx, valid_gt_idx, axis=0)
    ```
    
3.  找到匹配的`anchor`的最优`gt`，并为每个`anchor`的位置分配给其最匹配的`gt`
    
    ```python
    best_truth_overlap = tf.reduce_max(overlaps, 0)
    best_truth_idx = tf.argmax(overlaps, 0, tf.int32)
    best_truth_overlap = tf.tensor_scatter_nd_update(
        best_truth_overlap, best_prior_idx_filter[:, None],
        tf.ones_like(best_prior_idx_filter, tf.float32) * 2.)
    best_truth_idx = tf.tensor_scatter_nd_update(
        best_truth_idx, best_prior_idx[:, None],
        tf.range(tf.size(best_prior_idx), dtype=tf.int32))

    matches = tf.gather(bbox, best_truth_idx)
    ```
    
4.  设置每个`anchor`位置对应的置信度，如果其`iou score`小于阈值则设置为0
    
    
5.  编码生成`bbox label`，将`gt`中心坐标转换为相对`anchor`中心的偏移，且尺度除以移除以(方差*`gt`宽度)，`gt`的宽高除`anchor`宽高并进行对数化再除方差
    
    ```python
    label_loc = tf_encode_bbox(matches, self.anchors, self.variances)
    ```
    
6.  编码生成`landmark label`，计算`gt`中心坐标对应`anchor`的中心偏移且尺度除以移除以(方差*`anchor`宽度)
    
    ```python
    matches_landm = tf.gather(landm, best_truth_idx)
    label_landm = tf_encode_landm(matches_landm, self.anchors, self.variances)
    ```
    
7.  生成`calss label`，将无效的`anchor`位置概率设置为0
    
    ```python
    label_conf = tf.gather(clses, best_truth_idx)
    # filter gt and anchor overlap less than pos_thresh, set as background
    label_conf = tf.where(best_truth_overlap[:, None] < self.pos_thresh,
                              tf.zeros_like(label_conf), label_conf)
        ```

# 损失计算

1.  直接根据`calss label`得到`mask`
    
    ```python
    bc_num = tf.shape(y_pred)[0]
    loc_data, landm_data, conf_data = tf.split(y_pred, [4, 10, 2], -1)
    loc_t, landm_t, conf_t = tf.split(y_true, [4, 10, 1], -1)
    # landmark loss
    pos_landm_mask = tf.greater(conf_t, 0.)  # get valid landmark num
    ```
    
2.  根据`mask`得到有效的`landmark label`，直接使用`smooth_l1_loss`
    
    ```python
    num_pos_landm = tf.maximum(tf.reduce_sum(tf.cast(pos_landm_mask, tf.float32)), 1)  # sum pos landmark num
    pos_landm_mask = tf.tile(pos_landm_mask, [1, 1, 10])  # 10, 16800, 10
    # filter valid lanmark
    landm_p = tf.reshape(tf.boolean_mask(landm_data, pos_landm_mask), (-1, 10))
    landm_t = tf.reshape(tf.boolean_mask(landm_t, pos_landm_mask), (-1, 10))
    loss_landm = tf.reduce_sum(huber_loss(landm_t, landm_p))
    ```
    
3.  根据`mask`得到有效的`bbox label`，直接使用`smooth_l1_loss`
    
    ```python
    pos_loc_mask = tf.tile(pos_conf_mask, [1, 1, 4])
    loc_p = tf.reshape(tf.boolean_mask(loc_data, pos_loc_mask), (-1, 4))  # 792,4
    loc_t = tf.reshape(tf.boolean_mask(loc_t, pos_loc_mask), (-1, 4))
    loss_loc = tf.reduce_sum(huber_loss(loc_p, loc_t))
    ```
    
4.  利用`logsumexp`将预测出的分类概率求和，得到所有类别概率之和并**减去当前这个`anchor`所负责的类别的概率**。
    
    ```python
    # Compute max conf across batch for hard negative mining
    batch_conf = tf.reshape(conf_data, (-1, 2))  # 10,16800,2 -> 10*16800,2
    loss_conf = (tf.reduce_logsumexp(batch_conf, 1, True) -
                 tf.gather_nd(batch_conf,
                              tf.concat([tf.range(tf.shape(batch_conf)[0])[:, None],
                                         tf.reshape(conf_t, (-1, 1))], 1))[:, None])
    ```

5.  难例挖掘，根据`mask`得到所有的负样本概率，进行排序并选择合适的负样本数量。
    
    ```python
    loss_conf = loss_conf * tf.reshape(tf.cast(tf.logical_not(pos_conf_mask), tf.float32), (-1, 1))
    loss_conf = tf.reshape(loss_conf, (bc_num, -1))
    idx_rank = tf.argsort(tf.argsort(loss_conf, 1, direction='DESCENDING'), 1)

    num_pos_conf = tf.reduce_sum(tf.cast(pos_conf_mask, tf.float32), 1)
    num_neg_conf = tf.minimum(lsfn.negpos_ratio * num_pos_conf,
                              tf.cast(tf.shape(pos_conf_mask)[1], tf.float32) - 1.)
    neg_conf_mask = tf.less(tf.cast(idx_rank, tf.float32),
                            tf.tile(num_neg_conf, [1, tf.shape(pos_conf_mask)[1]]))[..., None]
    ```
6.  根据正样本的位置和难例挖掘负样本位置，得到需要计算概率值的位置，对这些位置计算其交叉熵。
    ```python
    # calc pos , neg confidence loss
    pos_idx = tf.tile(pos_conf_mask, [1, 1, 2])
    neg_idx = tf.tile(neg_conf_mask, [1, 1, 2])

    conf_p = tf.reshape(tf.boolean_mask(
        conf_data,
        tf.equal(tf.logical_or(pos_idx, neg_idx), True)), (-1, 2))
    conf_t = tf.boolean_mask(conf_t, tf.equal(tf.logical_or(pos_conf_mask, neg_conf_mask), True))

    loss_conf = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(conf_t, conf_p))
    ```

# 推理

推理好像没啥好说的

1.  得到输出概率，根据概率过滤得`bbox`和`landmark`
    
    ```python
    """ softmax class"""
    clses = softmax(clses, -1)
    score = clses[:, 1]
    """ decode """
    bbox = decode_bbox(bbox, h.anchors.numpy(), h.variances.numpy())
    bbox = bbox * np.tile(h.org_in_hw[::-1], [2])
    """ landmark """
    landm = decode_landm(landm, h.anchors.numpy(), h.variances.numpy())
    landm = landm * np.tile(h.org_in_hw[::-1], [5])
    """ filter low score """
    inds = np.where(score > obj_thresh)[0]
    bbox = bbox[inds]
    landm = landm[inds]
    score = score[inds] 
    ```
    
2.  解码`bbox`和`landmark`，然后`nms`就完事了

    
    ```python
    """ keep top-k before NMS """
    order = np.argsort(score)[::-1]
    bbox = bbox[order]
    landm = landm[order]
    score = score[order]
    """ do nms """
    keep = nms_oneclass(bbox, score, nms_thresh)
    bbox = bbox[keep]
    landm = landm[keep]
    score = score[keep]
    """ reverse img """
    bbox, landm = reverse_ann(bbox, landm, h.org_in_hw, np.array(orig_hw))
    results.append([bbox, landm, score])
    ```

# 总结

1.  总的来说`SSD`和`YOLO`的思想都很好，像`YOLO`中就没有难例挖掘的东西，因为他把所有负样本都计算损失了～

2.  不过我觉得`YOLO`中聚类`anchor`的方式放到`SSD`里面绝对是有效的，因为原本的`SSD`的`anchor`都是方形的，这样肯定效果没有特定比例的`anchor`效果好，并且我用聚类生成`anchor`之后，模型收敛明显加快了。

3.  还有就是`SSD`的模型没有上采样部分，这样速度虽然快，但是感受野就没法共享了，现在增加`SSD`模型的感受野的方式可以在模型中添加`FPN`、`SSH`、`RFB`模块。

4.  最后我最近还是对`DIOU`很感兴趣的，希望有空可以加上去试试效果。接下来一段时间得搞语音的东西发论文去了。
