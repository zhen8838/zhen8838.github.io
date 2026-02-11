---
title: OpenPose人体姿态估计
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-05-16 12:12:08
tags:
- 姿态估计
- Tensorflow
---

我参考`tf-pose-estimation`重新构建一个人体姿态估计模型,下面主要记录一下坑点.

<!--more-->


# 标签制作

1.  `tensorflow`重写`heatmap`失败.

这个姿态估计其实和`CenterNet`是差不多的实现思路,构建标签的时候都是通过对图像的对应位置放置高斯分布的方式制作出`heatmap`,我一开始看懂了之后就准备直接用`tensorflow`的方式重写,结果如下:
```python
  def get_heatmap(self, im_h, im_w, joint_list, th=4.6052, sigma=8.):

    heatmap: tf.Variable = tf.Variable(tf.zeros((self.hparams.parts, im_h, im_w)), trainable=False)

    for joints in joint_list:
      for i, center in enumerate(joints):
        if center[0] < 0 or center[1] < 0:
          continue

        delta = tf.sqrt(th * 2)
        # p0 -> x,y    p1 -> x,y
        im_wh = tf.cast((im_w, im_h), tf.float32)
        p0 = tf.cast(tf.maximum(0., center - delta * sigma), tf.int32)
        p1 = tf.cast(tf.minimum(im_wh, center + delta * sigma), tf.int32)

        x = tf.range(p0[0], p1[0])[None, :, None]
        y = tf.range(p0[1], p1[1])[:, None, None]

        p = tf.concat([x + tf.zeros_like(y), tf.zeros_like(x) + y], axis=-1)
        exp = tf.reduce_sum(tf.square(tf.cast(p, tf.float32) - center), -1) / (2. * sigma * sigma)
        # use indices update point area
        indices = tf.concat([tf.ones(p.shape[:-1] + [1], tf.int32) * i,
                             p[..., ::-1]], -1)
        # NOTE p is [x,y] , but `gather_nd` and `scatter_nd` require [y,x]
        old_center_area = tf.gather_nd(heatmap, indices)
        center_area = tf.minimum(tf.maximum(old_center_area, tf.exp(-exp)), 1.0)
        center_area = tf.where(exp > th, old_center_area, center_area)

        heatmap.scatter_nd_update(indices, center_area)
    # use indices update heatmap background NOTE scatter_nd can't use -1
    heatmap.scatter_update(tf.IndexedSlices(
        tf.clip_by_value(1. - tf.reduce_max(heatmap, axis=0), 0., 1.),
        self.hparams.parts - 1))

    heatmap_tensor = tf.transpose(heatmap, (1, 2, 0))

    if self.target_hw:
      heatmap_tensor = tf.image.resize(heatmap_tensor, self.target_hw)

    return heatmap_tensor
```

到这里其实是可以正确运行的,但是问题在于`joint_list`(关键点注释)他是个数是不确定的,所以还得把这个函数改成动态迭代的(用`tf.while_loop`),调这个函数我都调到吐了..所以我暂时先不搞全`tf`化.


2.  向量化制作`heatmap`

之前`tf`化输入管道太麻烦我就直接把他全部的输入管道复制过来,但是性能问题让我吐血...而且有个很奇特问题,我本来是用`tensor_slices`制作`tf.dataset`的,虽然速度慢但还是十几秒还是有一个`batch`.然后我改成`tfrecord`的方式,发现可能是前端读的数据太快,后面标签制作太慢反而导致一分钟才能读取一个`batch`...

然后我又得重写`numpy`版的制作`heatmap`.

```python
def get_heatmap_v(self, target_hw, joint_list_mask):
  heatmap = np.zeros((self.coco_parts, self.height, self.width), dtype=np.float32)
  height, width = self.height, self.width
  for (joints, masks) in zip(self.joint_list, joint_list_mask):
    for idx, (point, mask) in enumerate(zip(joints, masks)):
      if mask:
        th = 4.6052
        delta = np.sqrt(th * 2)
        p0 = np.maximum(0., point - delta * self.sigma).astype('int32')
        p1 = np.minimum([width, height], point + delta * self.sigma).astype('int32')

        x = np.arange(p0[0], p1[0])
        y = np.arange(p0[1], p1[1])

        xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

        exp = (((xv - point[0]) ** 2 + (yv - point[1]) ** 2) /
                (2.0 * self.sigma * self.sigma))

        yidx = yv[exp < th]
        xidx = xv[exp < th]
        exp_valid = exp[exp < th]

        heatmap[idx, yidx, xidx] = np.minimum(
            np.maximum(heatmap[idx, yidx, xidx],
                        np.exp(-exp_valid)), 1.0)

  heatmap = heatmap.transpose((1, 2, 0))

  # background
  heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

  if target_hw:
    heatmap = cv2.resize(heatmap, target_hw[::-1], interpolation=cv2.INTER_LINEAR)

  return heatmap

def get_heatmap(self, target_hw):
  heatmap = np.zeros((self.coco_parts, self.height, self.width), dtype=np.float32)

  for joints in self.joint_list:
    for idx, point in enumerate(joints):
      if point[0] < 0 or point[1] < 0:
        continue
      ImageMeta.put_heatmap(heatmap, idx, point, self.sigma)

  heatmap = heatmap.transpose((1, 2, 0))

  # background
  heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

  if target_hw:
    heatmap = cv2.resize(heatmap, target_hw[::-1], interpolation=cv2.INTER_LINEAR)

  return heatmap

@staticmethod
def put_heatmap(heatmap, plane_idx, center, sigma):
  center_x, center_y = center  # point
  _, height, width = heatmap.shape[:3]  # 热图大小

  th = 4.6052
  delta = math.sqrt(th * 2)

  x0 = int(max(0, center_x - delta * sigma))
  y0 = int(max(0, center_y - delta * sigma))

  x1 = int(min(width, center_x + delta * sigma))
  y1 = int(min(height, center_y + delta * sigma))

  cnt = 0
  uncnt = 0
  for y in range(y0, y1):
    for x in range(x0, x1):
      d = (x - center_x) ** 2 + (y - center_y) ** 2
      exp = d / 2.0 / sigma / sigma
      # NOTE 如果这个点不是靠近图像边沿,exp就是th的两倍
      if exp > th:
        continue
      heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
      heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)
```



然后看一下耗时对比,直接提高32倍~
```sh
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   666                                           @func_line_time([])
   667                                           def dev_vector_get_heatmap():
   668                                             """ 构造一个向量化的heatmap函数,并测试速度 """
   669         1     653201.0 653201.0      1.3    h, ds = get_raw_dataset()
   670                                             h: OpenPoseHelper
   671         1       8555.0   8555.0      0.0    iters = iter(ds)
   672       101        142.0      1.4      0.0    for i in range(100):
   673       100     319597.0   3196.0      0.6      img, joint_list = next(iters)
   674       100      12173.0    121.7      0.0      img, joint_list = img.numpy(), joint_list.numpy()
   675                                           
   676       100       3520.0     35.2      0.0      meta = ImageMeta(img, joint_list, h.in_hw, h.hparams.parts, h.hparams.vecs, h.hparams.sigma)
   677                                           
   678       100   48977247.0 489772.5     95.1      heatmap = meta.get_heatmap(h.target_hw)
   679       100       4721.0     47.2      0.0      joint_list_mask = np.logical_not(np.all(joint_list == -1000, -1, keepdims=True))
   680       100    1489782.0  14897.8      2.9      heatmap_v = meta.get_heatmap_v(h.target_hw, joint_list_mask)
   681                                           
   682       100      25532.0    255.3      0.0      print(np.allclose(heatmap, heatmap_v))
```

我发现我有时候就是太喜欢过早优化代码..这真的是万恶之源,很容易吃力不讨好.一开始直接用`numpy`重写就能顺利提高速度还不用费那么多时期.