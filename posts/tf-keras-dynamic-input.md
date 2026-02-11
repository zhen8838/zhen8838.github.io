---
title: tf.keras实现动态多尺度训练
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-11-13 21:36:27
tags:
-   Tensorflow
-   Yolo
-   Keras
---


哇,今天真的好累,就写了个动态多尺度训练(差点又被`tensorflow`劝退.),下面写几个要注意的点.

<!--more-->


# input_shape的修改

首先网络为了要支持多尺度训练,那我们的输入是不能指定的,所以要修改成$[None,None,3]$才可以.

# output_shape的修改

这里就是一个很蛋疼的地方,因为我们不知道输入的维度,所以得到的输出维度为$[None,None,75]$,之前我们可以计算出前两个维度然后`reshape`,但是现在我们不知道这个维度`reshape`中传入`None`就会报错.但是如果不`Reshape`,在`tf.keras`的`Loss`中会默认检查`y_true`和`y_pred`的维度是否一致,所以我只能用折中的方式,使用`expand_dims`加一个虚的维度,然后在`Loss`中重新`reshape`

# xy_offset的计算

我本来的想法是修改`Helper`对象中的输入输出数组值即可,但是改了之后我才发现,由于`tf.keras`用的是静态图的方式,所以我使用的那个数组早就在建立图的时候被转换为`Tensor`固化了,我再去修改他也并没有实际用处...所以得在`Loss`里面重新计算`xy_offset`了.

# 分别处理训练和测试

根据[测试tf.keras中callback的运行状态](https://zhen8838.github.io/2019/11/14/tf-keras-callback/)中测试的方式,添加`Callback`对象进行处理,这种方式唯一的缺点就是在测试开始之后,有一部分`batch`本来应该是多尺度的,但是被强制改成原始尺度了.哎,实在想不到好点子了.....日常劝退`tensorflow`.

# 总结

花了半天时间搞定了这个多尺度训练,接下来可以做别的去了...之前`lffd`算法的内存泄漏问题调试的我头疼,都没时间测试算法的效果.