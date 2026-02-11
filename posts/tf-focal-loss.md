---
title: bce focal loss
mathjap: true
categories:
  - 深度学习
date: 2020-02-16 14:33:29
tags:
-   Tensorflow
---

简单记录一下`bce focal loss`。


<!--more-->

# bce loss
公式为：
$$
\begin{aligned}
loss=&y * -\log(sigmoid(p)) + (1 - y) * -\log(1 - sigmoid(p))\\
    =&\begin{cases} -\log(\sigma(p))&,\ \ \ y=1\\
                    -\log(1-\sigma(p))&,\ \ \ y=0\\
     \end{cases}
\end{aligned}
$$


为了解决**正负样本的不平衡问题**，通常添加参数$\alpha$平衡损失：
$$
\begin{aligned}
    loss=&\begin{cases} -\alpha \log(\sigma(p))&,\ \ \ y=1\\
                    -(1-\alpha)\log(1-\sigma(p))&,\ \ \ y=0\\
     \end{cases}
\end{aligned}
$$


为了解决**样本的难易程度不平衡问题**，基于预测置信度平衡难易程度：

$$
\begin{aligned}
    loss=&\begin{cases} -(1-\sigma(p))^\gamma \log(\sigma(p))&,\ \ \ y=1\\
                    -\sigma(p)^\gamma\log(1-\sigma(p))&,\ \ \ y=0\\
     \end{cases}
\end{aligned}
$$

这样置信度越高的正样本和置信度高的负样本损失衰减都会较大。

# focal loss

结合类别不平衡问题与难易程度不平衡问题，得到`focal loss`：

$$
\begin{aligned}
    loss=&\begin{cases} -\alpha(1-\sigma(p))^\gamma \log(\sigma(p))&,\ \ \ y=1\\
                    -(1-\alpha)\sigma(p)^\gamma\log(1-\sigma(p))&,\ \ \ y=0\\
     \end{cases}
\end{aligned}
$$


```python
def focal_sigmoid_cross_entropy_with_logits(labels: tf.Tensor, logits: tf.Tensor,
                                            gamma: float = 2.0,
                                            alpha: float = 0.25):
    pred_sigmoid = tf.nn.sigmoid(logits)
    pt = (1 - pred_sigmoid) * labels + pred_sigmoid * (1 - labels)
    focal_weight = (alpha * labels + (1 - alpha) * (1 - labels)) * tf.math.pow(pt, gamma)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits) * focal_weight
    return loss
```