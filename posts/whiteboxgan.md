---
title: White-box GAN
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-10-09 15:25:41
tags:
- GAN
---

关于论文 Learning to Cartoonize Using White-box Cartoon Representations


<!--more-->

# 数据集

为了使得模型能对人像与景物生成均有较好的效果，训练时需要`[photo_face,photo_scenery,cartoon_face,cartoon_scenery]`四种类型的图像。他每采样5次风景图像采样一次人物图像。

# 损失函数

首先给出整体的流程图，有一个宏观的概念。

![](whiteboxgan/process.svg)

作者通过观察动画图像认为：

- 动画图像主要包括整体的结构特征

- 轮廓细节使用的清晰和锐化的线

- 平滑与平坦的表面颜色

因此提出三种损失函数

##  Learning From the Surface Representation

论文中提出要提取`Surface Representation`特征，即图像的平滑表面特征。作者通过调研使用`differentiable guided filter`对图像进行平滑处理。

**NOTE：** 这里滤波器方法的原论文为`Fast end-to-end trainable guided filter`，但作者在使用时将滤波器权重进行了固定。

定义滤波器为$\mathcal{F}_{dgf}$，输入真实图像为$\mathcal{I}_p$，输入动画图像为$\mathcal{I}_c$，将生成图像的滤波结果与卡通图像的滤波结果进行判别器判别。

$$
\begin{aligned}
\mathcal{L}_{\text {surface}}\left(G, D_{s}\right) &=\log D_{s}\left(\mathcal{F}_{\text {dgf}}\left(\boldsymbol{I}_{c}, \boldsymbol{I}_{c}\right)\right) \\
&+\log \left(1-D_{s}\left(\mathcal{F}_{\text {dgf}}\left(G\left(\boldsymbol{I}_{p}\right), G\left(\boldsymbol{I}_{p}\right)\right)\right)\right)
\end{aligned}\tag{1}
$$


##   Learning From the Structure representation

这点我觉得比较好，作者想到`superpixel`后的图像具备较大的图像块以及清晰的边界，从结果上来看`superpixel`后的图像已经初步具备动画图像的一些特征了，因此很时候来学习原始图像的结构特征。作者在原始的的`superpixel`基础上加入了选择性搜索来合并一些超像素区域，这样可以获得更加大块的像素，并且考虑到传统的超像素算法是利用平均的方法来合成大像素的，对于图像将会降低部分的亮度与对比度，因此再提出自适应着色算法，提升`superpixel`的对比度。

对于结构特征的损失实际上是预训练模型的编码差异损失：
$$
\begin{aligned}
\mathcal{L}_{\text {structure}}=\left\|\operatorname{VGG}_{n}\left(G\left(\boldsymbol{I}_{p}\right)\right)-\operatorname{VGG}_{n}\left(\mathcal{F}_{\text {st}}\left(G\left(\boldsymbol{I}_{p}\right)\right)\right)\right\|
\end{aligned}\tag{2}
$$


##  Learning From the Textural Representation

学习纹理特征，考虑到判别器可以通过颜色亮度等特征很容易的区别出真实图像与动画图像，因此使用灰度化的图像去除其颜色信息，提取其单通道的纹理特征来进行判别。
$$
\begin{aligned}
  \mathcal{F}_{r c s}\left(\boldsymbol{I}_{r g b}\right)=(1-\alpha)\left(\beta_{1} * \boldsymbol{I}_{r}+\beta_2 * \boldsymbol{I}_{g}+\beta_{3} * \boldsymbol{I}_{b}\right)+\alpha * \boldsymbol{Y}
\end{aligned}\tag{3}
$$

**NOTE:** 这里和我之前学习过得`AnimeGan`有相似之处，不过这里的灰度化使用的是随机灰度化，即$\beta$的值是从均匀分布中选取的。值得一提的是论文提到的$\alpha$在开源代码中并没有出现，作者直接将均匀分布的范围控制在$(0,1-\alpha)$完成了相同的目标。

## Image up-sampling using total-variation regularization with a new observation model.


作者使用此论文中的`total-variation`损失强制生成结果更加平滑：
$$
\begin{aligned}
  \mathcal{L}_{t v}=\frac{1}{H * W * C}\left\|\nabla_{x}\left(G\left(\boldsymbol{I}_{p}\right)\right)+\nabla_{y}\left(G\left(\boldsymbol{I}_{p}\right)\right)\right\|
\end{aligned}\tag{4}
$$

## 内容一致性损失

这里也和`AnimeGan`有相似之处，不过`AnimeGan`里还使用了生成图像和当前动画图像的`style loss`。

$$
\begin{aligned}
  \mathcal{L}_{\text {content}}=\left\|\operatorname{VGG}_{n}\left(G\left(\boldsymbol{I}_{p}\right)\right)-\operatorname{VGG}_{n}\left(\boldsymbol{I}_{p}\right)\right\|
\end{aligned}\tag{5}
$$

## 一些问题

我发现原论文开源代码中，貌似$G\left(\boldsymbol{I}_{p}\right)$其实都被替换成了$\mathcal{F}_{\text {st}}\left(G\left(\boldsymbol{I}_{p}\right)\right)$，先问问论文作者看看。