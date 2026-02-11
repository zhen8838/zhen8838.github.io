---
title: PFLD总结
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-12-21 12:09:02
tags:
-   Tensorflow
-   人脸检测
---

[PFLD](https://arxiv.org/pdf/1902.10859.pdf)算法是我今年8月复现的,当时没有写总结,现在补上.


<!--more-->


# 网络设计

&emsp;&emsp;`PFLD`的模型相对简单,但是想法的确挺不错的.首先使用一个`mobilenet`或者别的网络做`backbone`,和传统的目标检测模型相同,抽取三个不同层次的`feature map`并汇总作为`landmark`输出,并且从网络中再抽取一个`feature map`用于预测`euler angles`.


# 标签制作

这里的制作`label`时会计算当前整个`batch`的样本数量,并根据人脸的属性类别去计算属性权重`attribute_weight`,然后就是归一化的`landmark`与`euler angles`.


# 损失计算

标签中包含`landmark`、`euler angles`、`attribute weight`，预测结果包含`pred_landmark`、`pred euler angles`。对于`landmark`的回归直接`mse`，对于`euler angles`的回归为$1-\cos(\text{abs}(true\ eular-pred\ eular))$

# 推理

推理就直接把所有`landmark`乘上输入图像大小就好了。