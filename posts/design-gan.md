---
title: Design GAN
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-11-09 11:58:15
tags:
- GAN
---

`DESIGN-GAN: CROSS-CATEGORY FASHION TRANSLATION DRIVENBY LANDMARK ATTENTION`这是来自Alibaba的一篇论文，不过他投的会议，一共只有5页，感觉有的部分没有说清楚。这篇论文提出一种基于landmark 引导的注意力cyclegan，用于人物换装。


<!--more-->

# 创新点

传统的基于cyclegan方法通常只能对形状匹配的物体进行生成，而对于形状不同的裙子-裤子难以进行跨域的生成，后续提出来的一些方法虽然可以跨域的生成，但纹理难以控制，长裤变短裤之后的皮肤颜色就很不自然。

1.  提出DesignGAN方法，基于landmark attention引导的服饰变化方法
2.  利用纹理相似度限制机制对纹理进行生成
3.  不但可以保持纹理，还可以利用额外的纹理的图像修改对应服饰的纹理。


# 架构

![](design-gan/design-gan-arch.png)

## 生成器部分


1.  **利用基于HR-Net骨干的landmark回归网络生成landmark heatmap**
    文章说训练了两个模型，一个检测人体的关键点，一个检测服饰的关键点。然后通过反卷积生成多个通道的heatmap，一个通道对应一个landmark点，利用mseloss训练整个landmark heatmap生成器。
   
2.  **利用特征提取器对服饰图像与原始图像提取服饰特征**

    原文中说服饰图像是利用landmark回归网络生成landmark attention引导服饰区域，但是没有说明是怎么样引导的，我猜测是用物体的landmark将图像的区域进行连接，然后生成mask图像。    
    特征提取器实际上应该是有两个，一个对原图进行提取。然后利用landmark生成的mask来提取特征。

3.  **特征concat之后进行生成**

    这里我认为他的实现上是两个独立的生成器或者中间带特征融合的多输出生成器，一个生成目标图像，另一个生成目标服饰的mask。

## 判别器部分

1.  **特征提取**
    用相同的特征提取器进行特征提取

2.  **利用landmark对人体特征进行加权**

    回归出来的landmark heatmap原文说是尺度与原图像相同，这里又使用逐元素积加权，那么推测特征提取器的输出特征应该是3维的，甚至可能和原图大小也一样。

3.  **特征融合后判别器判别**


# 损失

## CycleGAN loss

还是几个基本的loss，lsgan loss+循环一致性 loss+身份映射 loss：

$$
\begin{aligned}
\mathcal{L}_{L S G A N}=& \mathbb{E}_{(y, \mathbf{b}) \sim p_{\text {data}}}\left[\left(D_{Y}(y, \mathbf{b})-1\right)^{2}\right]+\\
& \mathbb{E}_{(x, \mathbf{a}) \sim p_{\text {data}}}\left[D_{Y}\left(G_{X Y}(x, \mathbf{a})\right)^{2}\right]
\end{aligned}
$$

$$
\begin{array}{r}
\mathcal{L}_{c y c}=\mathbb{E}_{(x, \mathbf{a}) \sim p_{\text {data}}}\left[\left\|G_{Y X}\left(G_{X Y}(x, \mathbf{a})\right)-(x, \mathbf{a})\right\|_{1}\right]+ \\
\mathbb{E}_{(y, \mathbf{b}) \sim p_{\text {data}}}\left[\left\|G_{X Y}\left(G_{Y X}(y, \mathbf{b})\right)-(y, \mathbf{b})\right\|_{1}\right]
\end{array}
$$

$$
\begin{array}{r}
\mathcal{L}_{i d t}=\mathbb{E}_{(y, \mathbf{b}) \sim p_{\text {data}}}\left[\left\|G_{X Y}(y, \mathbf{b})-(y, \mathbf{b})\right\|_{1}\right]+ \\
\mathbb{E}_{(x, \mathbf{a}) \sim p_{\text {data}}}\left[\left\|G_{Y X}(x, \mathbf{a})-(x, \mathbf{a})\right\|_{1}\right] .
\end{array}
$$


## 纹理损失和皮肤损失


对于生成前后服饰区域的图像，统一resize到相同大小，然后直接计算rgb的像素差异作为损失。这里有个比较好的点就是提出$w_{\text {style}}$，应该类似center loss中的高斯heatmap的反向，越中心的权重越低，越边缘的区域权重越高，因为边缘区域的细节更加重要（比如裙子和背景的交界处）


$$
\mathcal{L}_{\text {style}}=\frac{1}{2 N} \sum_{n=1}^{N} \sum_{c=1}^{3}\left\|p(\mathbf{a})_{i, j}-p\left(\mathbf{b}^{\prime}\right)_{i, j}\right\|_{2} \circ w_{\text {style}}
$$


皮肤损失和上面类似，这里是使用了预先准备的皮肤纹理图像进行与原始图像中的皮肤进行匹配（根据landmark提取手臂关节处的皮肤，但文章没写穿长袖的时候怎么办。。）

$$
\mathcal{L}_{s k i n}=\frac{1}{2 N} \sum_{n=1}^{N} \sum_{c=1}^{3}\left\|p\left(\mathbf{a}_{s}\right)_{i, j}-p\left(\mathbf{b}_{s}^{\prime}\right)_{i, j}\right\|_{2} \circ w_{s k i n}
$$


最后多个损失合成一个损失。

$$
\begin{array}{r}
\mathcal{L}_{A l l}=\mathcal{L}_{L S G A N}+\lambda_{c y c} \mathcal{L}_{c y c}+\lambda_{i d t} \mathcal{L}_{i d t}+ \\
\lambda_{s t y l e} \mathcal{L}_{s t y l e}+\lambda_{s k i n} \mathcal{L}_{s k i n}
\end{array}
$$

## 纹理定制

直接将纹理图像填充到上面描述的目标服饰的mask中，然后利用纹理损失就可以进行定制纹理了。

# 总结

因为篇幅限制，这篇文章细节讲的属实有点少。我最想知道他的特征提取器是什么架构，提取出来的特征维度到底是怎样的，但是他文章中说特征提取器不依赖具体架构。。。