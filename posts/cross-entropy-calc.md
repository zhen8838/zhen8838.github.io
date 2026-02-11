---
title: Cross Entropy的数值稳定计算
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-09-22 16:52:41
tags:
- Tensorflow
- 损失函数
---


今天在看`centernet`的`heatmap`损失函数时,发现他的损失和熵差不多,但是我用`tf`的实现会导致`loss`为`Nan`,因此我看了下`Cross Entropy`的计算优化,这里记录一下.

<!--more-->

## Tensorflow中的cross_entropy计算

令$x = logits$,$z = labels$:
$$
\begin{aligned}
    &  z * -\log(\text{sigmoid}(x)) + (1 - z) * -\log(1 - \text{sigmoid}(x)) \\
=& z * -\log(\frac{1}{1 + e^{-x}}) + (1 - z) * -\log(\frac{e^{-x}}{1 + e^{-x}}) \\
=& z * \log(1 + e^{-x}) + (1 - z) * (-\log(e^{-x}) + \log(1 + e^{-x})) \\
=& z * \log(1 + e^{-x}) + (1 - z) * (x + \log(1 + e^{-x}) \\
=& (1 - z) * x + \log(1 + e^{-x}) \\
=& x - x * z + \log(1 + e^{-x}) \\
=& \log(e^x) - x * z + \log(1 + e^{-x}) \\
=& - x * z + \log(1 + e^{x})
\end{aligned}
$$

下面为了避免$e^{x}$数值溢出,因此优化为如下:

$$
\begin{aligned}
  &  \log(1 + e^{x}) \\
=&  \log(1 + e^{-|x|}) + \max(x, 0)
\end{aligned}   
$$

**NOTE:** `tensorflow`中有个专门的函数$softplus(x)=\log(1 + e^{x})$,其中已经包含了数值溢出的优化.


## Centernet中的FocalLoss计算

先给出他的`FocalLoss`部分代码:

```python
def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred [batch,c,h,w]
      gt_regr [batch,c,h,w]
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss
```

**NOTE:** 注意这里的`pred`是经过`sigmoid`的.

将上述代码转换为公式,令$x = logits$,$z = labels$,$x_s=\text{sigmoid}(x)$:
$$
\begin{aligned}
  & -\log(\text{sigmoid}(x))*(1-x_s)^2-\log(1-\text{sigmoid}(x))* x_s^2\\
= & -\log(\frac{1}{1+e^{-x}})*(1-x_s)^2-\log(\frac{e^{-x}}{1+e^{-x}})* x_s^2\\
= & \log(1+e^{-x})*(1-x_s)^2+[-\log(e^{-x}) + \log(1 + e^{-x})]*x_s^2] \\
= & \text{softplus}(-x)*(1-x_s)^2+[x + \text{softplus}(-x)]*x_s^2]
\end{aligned}   
$$

优化后对应代码为:
```python
  def focal_loss(self, true_hm: tf.Tensor, pred_hm: tf.Tensor) -> tf.Tensor:
      """ Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory

      Parameters
      ----------
      true_hm : tf.Tensor
          shape : [batch, out_h , out_w, calss_num]
      pred_hm : tf.Tensor
          shape : [batch, out_h , out_w, calss_num]

      Returns
      -------
      tf.Tensor
          heatmap loss
          shape : [batch,]
      """
      z = true_hm
      x = pred_hm
      x_s = tf.sigmoid(pred_hm)

      pos_inds = tf.cast(tf.equal(z, 1.), tf.float32)
      neg_inds = 1 - pos_inds
      neg_weights = tf.pow(1 - z, 4)

      # neg entropy loss =  −log(sigmoid(x)) ∗ (1−sigmoid(x))^2 − log(1−sigmoid(x)) ∗ sigmoid(x)^2
      loss = tf.add(tf.nn.softplus(-x) * tf.pow(1 - x_s, 2) * pos_inds, (x + tf.nn.softplus(-x)) * tf.pow(x_s, 2) * neg_weights * neg_inds)

      num_pos = tf.reduce_sum(pos_inds, [1, 2, 3])
      loss = tf.reduce_sum(loss, [1, 2, 3])

      return tf.div_no_nan(loss, num_pos)
```