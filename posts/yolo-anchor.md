---
title: yolo中anchor值的解释
categories:
  - 深度学习
date: 2019-03-12 12:17:54
mathjax: true
tags:
-   Tensorflow
-   Yolo
-   目标检测
---

`anchor`意味着先验思想，是为了给神经网络拟合`box`减轻负担。下面就来讲解一下`anchor`的计算。

<!--more-->

# 公式

首先我们得先理解`yolo`输出`boundary box`的计算过程：
$$ 
\begin{aligned}
    b_x&=\sigma(t_x)+c_x \\
    b_y&=\sigma(t_y)+c_y \\
    b_w&=p_w e^{t_w} \\
    b_h&=p_h e^{t_h} 
\end{aligned}
$$
解释：
$t_x,t_y,t_w,t_h$为`yolo`预测输出结果。

$c_x,c_y$为当前`cell`右上角的坐标。

$p_w,p_h$是当前`anchor`的宽高。

$b_x,b_y,b_w,b_h$则是最终`yolo`预测出的`boundary box`。

![](yolo-anchor/2.jpeg)

# 分析

通过上图我们知道,实际上`anchor`存在的意义就在于调节网络输出与真实`box`的比例关系.

制作`label`的过程:
$$ \begin{aligned}
Label_w&= \frac{True_w}{p_w} \\
Label_h&= \frac{True_h}{p_h} \\
\end{aligned} $$

然后我们训练的时候就是拟合$e^{t_w}$到$Label_w$
$$\begin{aligned}
   \because b_w&=p_w e^{t_w} \\
            b_h&=p_h e^{t_h}  \\
  \therefore
            e^{t_w} &\rightarrow Label_w \\
            e^{t_h} &\rightarrow Label_h  \\
\end{aligned} $$


所以通过设置合适的$p_w,p_h$值,我们可以把$Label_w,Label_h$控制在$1$左右.这样使得神经网络只需要在预测$w,h$时只需要接近$1$就可以取得较好的效果.

# 制作anchor

知道了原理,我们就可以来选取合适的`anchor`,我这里是自己写了个`kmeans`,然后将其中的距离计算改成了`iou`的函数:

![](yolo-anchor/3.png)

现在我们加载了自己的`anchor list`,然后测试一下:

```python
def test_label_wh():
    gl = helper.generator(is_make_lable=True, is_training=False)
    for i in range(20):
        imgl, label = next(gl)
        print("w_max: {:.3f} ,w_min: {:.3f} ,h_max: {:.3f} ,h_min: {:.3f}".format(
            np.max(label[np.where(label[..., 4] > .7)][:, 2]),
            np.min(label[np.where(label[..., 4] > .7)][:, 2]),
            np.max(label[np.where(label[..., 4] > .7)][:, 3]),
            np.min(label[np.where(label[..., 4] > .7)][:, 3])))
```
输出:

```sh
w_max: 1.159 ,w_min: 1.159 ,h_max: 0.906 ,h_min: 0.906
w_max: 0.944 ,w_min: 0.944 ,h_max: 1.223 ,h_min: 1.223
w_max: 1.299 ,w_min: 1.055 ,h_max: 0.864 ,h_min: 0.751
w_max: 0.939 ,w_min: 0.939 ,h_max: 1.294 ,h_min: 1.294
w_max: 0.992 ,w_min: 0.918 ,h_max: 1.289 ,h_min: 1.257
w_max: 1.346 ,w_min: 1.346 ,h_max: 0.965 ,h_min: 0.965
w_max: 1.225 ,w_min: 0.986 ,h_max: 0.905 ,h_min: 0.780
w_max: 1.139 ,w_min: 0.851 ,h_max: 0.961 ,h_min: 0.939
w_max: 0.979 ,w_min: 0.979 ,h_max: 0.941 ,h_min: 0.941
w_max: 1.062 ,w_min: 1.062 ,h_max: 0.957 ,h_min: 0.957
w_max: 1.399 ,w_min: 1.399 ,h_max: 0.831 ,h_min: 0.831
w_max: 0.945 ,w_min: 0.729 ,h_max: 1.102 ,h_min: 0.744
w_max: 0.921 ,w_min: 0.921 ,h_max: 0.867 ,h_min: 0.867
w_max: 0.854 ,w_min: 0.668 ,h_max: 1.298 ,h_min: 1.039
w_max: 1.060 ,w_min: 1.060 ,h_max: 0.648 ,h_min: 0.648
w_max: 0.972 ,w_min: 0.972 ,h_max: 0.925 ,h_min: 0.925
w_max: 0.805 ,w_min: 0.805 ,h_max: 1.126 ,h_min: 1.126
w_max: 0.809 ,w_min: 0.809 ,h_max: 0.958 ,h_min: 0.958
w_max: 1.134 ,w_min: 1.027 ,h_max: 1.502 ,h_min: 1.429
w_max: 1.629 ,w_min: 0.915 ,h_max: 1.208 ,h_min: 0.690
```

我们的`label`中$w,h$值都在$1$左右了,说明这个`anchor list`是合适的~