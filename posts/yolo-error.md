---
title: 实现yolo时踩过的坑！
categories:
  - 深度学习
date: 2019-07-10 15:15:58
mathjax: true
tags:
-   Tensorflow
-   Yolo
-   目标检测
---

终于把[yolo v3框架](https://github.com/zhen8838/K210_Yolo_framework)写好了。支持多模型、多数据集、任意输出层数量、任意anchor数量、模型剪枝还适配k210.不要太好用～

这里记录一下我之前的实现的问题出在哪里。
<!--more-->

# 错误地计算了ignore mask

从`yolo v2`开始就会计算正确`box`与预测`box`直接`iou`关系，如果`iou score`大于阈值，那么说明这个预测`box`是成功预测到了这个对象的，极大的提高了模型的`recall`。
但是我在开源的`yolo v2`中使用`Boolean mask`函数时忽略了一点，比如`batch`为16，那么输出`label`的尺寸为$[16,7,10,3,25]$,直接使用`Boolean mask`会得到正确`box`尺寸为$[?,4]$。然后我把这个$[?,4]$与预测出来的`box`$[16,7,10,3,4]$计算`iou score`。

乍一看还以为没什么毛病，其实这里最大的毛病就是整个`batch`的正确`box`都与整个`batch`的预测`box`都做了`iou score`，如果这时候计算最优`iou score`，很有可能这个最优的预测`box`不属于这张图片！数据直接出现了混合，这就是根源问题。

在新的写的代码中，我用了`map`的方式处理每张图片，既提高了效率，又避免了错误。

题外话一句。。我现在都惊讶我之前的`yolo v2`为啥效果还行，很有可能误打误撞搞了个数据集`mix`的效果。。。


**更新:**  通过对比腾讯优图所开源的`yolo3`的代码,我发现这个`ignore mask`不但需要每张图像单独计算,还需要单一输出层与全局的目标进行计算,因为我用的是`tf.keras`,所以没办法在不使用`hack`的方式下传入整张图像的`bbox`数组,所以我在`label`中多加了一维,标记全局的对象位置.

以下代码为我目前的标签制作代码:

* 避免了`inf` 
* 避免了对象重叠(原版yolo也没有考虑到这一点)
* 添加了全局的对象标记. 
  
这些问题消除之后,我的`yolo`所计算出的`loss`与腾讯优图所开源的`yolo`完全一致.终于完美复现出`yolo`的效果了~

```python
labels = [np.zeros((self.out_hw[i][0], self.out_hw[i][1], len(self.anchors[i]),
                    5 + self.class_num + 1), dtype='float32') for i in range(self.output_number)]

layer_idx, anchor_idx = self._get_anchor_index(ann[:, 3:5])
for box, l, n in zip(ann, layer_idx, anchor_idx):
    # NOTE box [x y w h] are relative to the size of the entire image [0~1]
    # clip box avoid width or heigh == 0 ====> loss = inf
    bb = np.clip(box[1:5], 1e-8, 0.99999999)
    cnt = np.zeros(self.output_number, np.bool)  # assigned flag
    for i in range(len(l)):
        x, y = self._xy_grid_index(bb[0:2], l[i])  # [x index , y index]
        if cnt[l[i]] or labels[l[i]][y, x, n[i], 4] == 1.:
            # 1. when this output layer already have ground truth, skip
            # 2. when this grid already being assigned, skip
            continue
        labels[l[i]][y, x, n[i], 0:4] = bb
        labels[l[i]][y, x, n[i], 4] = (0. if cnt.any() else 1.)
        labels[l[i]][y, x, n[i], 5 + int(box[0])] = 1.
        labels[l[i]][y, x, n[i], -1] = 1.  # set gt flag = 1
        cnt[l[i]] = True  # output layer ground truth flag
        if cnt.all():
            # when all output layer have ground truth, exit
            break
```


# anchor的尺度

前面我有个文章也写了，`anchor`的作用就是让预测`wh`与真实`wh`直接的比例接近与1，那么细细想来，`anchor`的尺度是对应图片尺度$[224,320]$还是对应栅格的尺度，还是对应全局的`0-1`都没有什么关系，只不过`anchor`的尺度就代表做标签的时候`label`要转换的尺度。所以为了方便起见，直接把`anchor`尺度设置为全局的`0-1`就完事了，还减少运算量。


# loss出现NaN

问题原因在于图片标签的`width`与`height`出现了`0`,导致`log(0)=-inf`的问题.
解决起来很简单,在制作标签的时候限制`width`与`height`范围即可.


# label中的极端情况的考虑

## bbox到达边界值

当`bbox`的中心点位于边界值最大值时,如下图所示.
$$\begin{aligned}
    index&=floor(x*w) \\
    \because w&=3,x=1 \Rightarrow floor(1*3)=3
\end{aligned}
$$
但使用`3`进行索引就会报错,所以我们需要限制一下`bbox`的中心坐标不能大于等于$1$.

    +-------+-------+-------+
    |       |       |       |
    |       |       |       |
    |       |       |  +---------+
    +-------+-------+--|----+    |
    |       |       |  |    |    |
    |       |       |  |  center |
    |       |       |  |    |    |
    +-------+-------+--|----+    |
    |       |       |  +---------+
    |       |       |       |
    |       |       |       |
    +-------+-------+-------+
                
## 当两个目标的label相同时

如下图所示,当两个`bbox`真的非常靠近时,就会出现他们的`label`所在的位置都是相同的,就会出现`label`被覆盖的问题了.目前我将相同`label`时,后面的`label`分配给次优的`anchor`.

    +---------------+-------+
    |  +---------+  |       |
    | +|--------+|  |       |
    | ||    |   ||  |       |
    +-||--------||----------+
    | ||    |   ||  |       |
    | ||    |   ||  |       |
    | ||    |   ||  |       |
    +-|+---------+----------+
    | +---------+   |       |
    |       |       |       |
    |       |       |       |
    +-------+-------+-------+
    
# 数据增强

数据增强我使用`gluoncv`的方式，首先是图像`crop`与`resize`，使用的是`ssd`所提出的带`iou`约束的`crop`方式，`resize`之后结合`imgaug`库进行数据增强，效果不错。如果可以再进一步，可以使用谷歌提出的`autoaugment`策略。我这里暂时还没用`mixup`，`gluoncv`里面应该是有使用的。

# IOULoss

推荐使用`ciou loss`，我测试之后`map`提高了4个点，效果相当不错。几个`iou loss`的实现方式我总结在[这里](https://zhen8838.github.io/2020/01/25/tf-ious/)

