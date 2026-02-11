---
title: 带宽受限下的DSA后端Compute Schedule
mathjax: true
toc: true
categories:
  - 编译器
date: 2023-02-23 23:28:48
tags:
- 后端优化
---

之前写过一篇带宽受限下的DSA后端优化, 不过主要是针对已经构建好Compute Schedule之后的优化, 今天准备展开讲讲. 
从单层卷积到优化计算,再到Layer Fusion,以及后续各种优化,下面将通过一系列的例子来介绍:

<!--more-->

# 0. 准备工作

首先需要实现高层IR的Index Mapping进行Infer Bounds, 这里我导入一个已经实现好的卷积的BoundsInfer. 

```python
from TracedArray import TarcedArray, GlobalHierarchy
from Conv2dBoundsInfer import Conv2dBoundsInfer, Segments
import torch
import numpy as np

Infer = Conv2dBoundsInfer(in_channels=2048, out_channels=512,
    kernel_size=1, groups=1, bias=True, padding=(0, 0),
    stride=(1, 1), dilation=(1, 1), intput_shape=(1, 2048, 56, 56),
    test=False)
```


# 1. 最简单的卷积实现
 
假设我们的DSA有一个比较大的SRAM, 并且可以在这个SRAM上执行Tensor级别的操作, 约定好SRAM大小为`L2SIZE`. 这里引入GlobalHierarchy作为多级内存存储抽象,用于计算数据加载次数, 检查存储是否溢出.
那么考虑在上面编写一个最Navie的卷积. 为了匹配Tensor级别的计算操作, 我们将原本按1进行for循环执行的逻辑看作为按tile大小为1取tensor进行计算.

```python
L2SIZE = 1536 * 1024 #

def demo1(imageArr: np.ndarray, weightArr: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  weight = TarcedArray(weightArr)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  (tileB, tileOC, tileOH, tileOW) = (1, 1, 1, 1)
  for b in Segments(0, B, tileB):
    for oc in Segments(0, OC, tileOC):
      for oh in Segments(0, OH, tileOH):
        for ow in Segments(0, OW, tileOW):
          with GlobalHierarchy(L2SIZE):
            # 进入SRAM之后 从DDR中加在数据并计算.
            outputTile = output[b, oc, oh, ow]
            imageTile = image[Infer.get_input_segment(b, oc, oh, ow)]
            weightTile = weight[Infer.get_w_segment(oc)]
            outputTile += np.sum((imageTile * weightTile), keepdims=True)

  assert (np.allclose(output._array, targetOutput, atol=1e-5))

  print("demo1 total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()


demo1 total loaded : 6578274304

```


# 2. 尝试进行硬件加速

可以发现我们重复加载了许多数据,也并没有加速计算, 但是实际上芯片中存在加速计算的硬件, 所以用以下方法来加速计算.

1. SRAM有足够空间的情况下, 可以尝试一次计算更大的tensor,也就是选择更大的tile size. 比如把W上的tile size设置为最大, 把H上tile size加大.

2. 假设我们有一个并行计算卷积部分输出的TensorCore, 一次最大并行输入16个input channel, 并行输出24个output channel.

接下来就可以来改造compute schedule:

```python
CORE_OC = 24 # TensorCore并行限制
CORE_IC = 16

def TensorCore(image: np.ndarray, weight: np.ndarray) -> np.ndarray:
  # 这里假设硬件可以自动循环
  return torch.conv2d(torch.tensor(image), torch.tensor(weight)).numpy()


def demo2(imageArr: np.ndarray, weightArr: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  weight = TarcedArray(weightArr)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  for b in Segments(0, B, 1):
    for oc in Segments(0, OC, CORE_OC):
      for oh in Segments(0, OH, 2):
        for ow in Segments(0, OW, OW):
          with GlobalHierarchy(L2SIZE):
            outputTile = output[b, oc, oh, ow]
            imageTile = image[Infer.get_input_segment(b, oc, oh, ow)]
            weightTile = weight[Infer.get_w_segment(oc)]
            outputTile += TensorCore(imageTile, weightTile)

  assert (np.allclose(output._array, targetOutput, atol=1e-5))

  print("demo2 total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()

demo2 total loaded : 179651584
```
    

# 3. 尝试减少重复的数据加载

可以发现demo2减少了许多数据加载, 但从事高性能计算的朋友们应该可以发现对weight来说, 如果在OH/OW有切分, 那么在OH/OW循环内都是每次加载相同的weights[ic,kh,kw]. 那么我们就有两个选择来解决这个问题:

1. 把weights加载的时机移动到OC循环内部或OC循环外部去加载, 这样在OH/OW的循环中可以不用load重复的weights了.

2. 我们还可以增加OH/OW的tile size,然后再添加一个IC的切分维度, 这样每个循环也不会重复加载weights了, 但是值得注意的是此时output tile需要移动到oc循环内.

## 3.1 尝试将Weights Stage到OC循环外

这里就是在SRAM中保存所有的weights, 实际上在stage到OC循环外之后, 我们还可以选择在OC循环内逐步的加载weights以进行流水. 

```python
def demo3_1(imageArr: np.ndarray, weightArr: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  weight = TarcedArray(weightArr)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  for b in Segments(0, B, 1):
    with GlobalHierarchy(L2SIZE):
      reuse = False # 为了简单起见, 添加reuse的参数来避免重复统计load的数据, 其实应该把allocate buffer和load/store的逻辑分离出来.
      weightTile = weight[Infer.get_w_segment(slice(0, OC))]  # 将weights加载移动到OC循环外 也就是一次加载所有的权重
      for oc in Segments(0, OC, CORE_OC):
        for oh in Segments(0, OH, 2):
          for ow in Segments(0, OW, OW):
            outputTile = output[b, oc, oh, ow]
            imageTile = image[Infer.get_input_segment(b, oc, oh, ow), reuse]  # 重用同一份SRAM
            if not reuse:
              reuse = True
            weightSubTile = weightTile[oc]
            outputTile += TensorCore(imageTile, weightSubTile)

  assert (np.allclose(output._array, targetOutput, atol=1e-5))
  print("demo3-1 total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()

demo3-1 We can't move load weights statement out of OC!
```


不过实际情况会发现因SRAM空间不够而出错. 这就是SRAM大小影响compute schedule.


## 3.2 尝试将Weights Stage到OC循环内

把weights stage在OC循环内部, 这样可以在OH/OW循环中复用. 

```python
def demo3_2(imageArr: np.ndarray, weightArr: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  weight = TarcedArray(weightArr)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  for b in Segments(0, B, 1):
    for oc in Segments(0, OC, CORE_OC):
      with GlobalHierarchy(L2SIZE):
        reuse = False
        weightTile = weight[Infer.get_w_segment(oc)]
        for oh in Segments(0, OH, 2):
          for ow in Segments(0, OW, OW):
            outputTile = output[b, oc, oh, ow]
            imageTile = image[Infer.get_input_segment(b, oc, oh, ow), reuse]  # 重用同一份SRAM
            if not reuse:
              reuse = True
            outputTile += TensorCore(imageTile, weightTile)

  assert (np.allclose(output._array, targetOutput, atol=1e-5))
  print("demo3-2 total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()

demo3-2 total loaded : 9586432


```


虽然这个例子和`demo3-1`实际上差别不大, 但是主要是用于说明SRAM对于不同的Compute Schedule的限制.

## 3.3 尝试新的切分维度

注意这里我们添加一个IC的切分维度, 这样每次内部循环加载的就是不同的weights[oc,ic,:,:]的tile了.

```python
def demo3_3(imageArr: np.ndarray, weightArr: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray, prefix="demo3-3"):
  image = TarcedArray(imageArr)
  weight = TarcedArray(weightArr)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  IC = imageArr.shape[1]
  for b in Segments(0, B, 1):
    for oc in Segments(0, OC, 8):
      for oh in Segments(0, OH, OH):
        for ow in Segments(0, OW, OW):
          with GlobalHierarchy(L2SIZE):
            reuse = False
            outputTile = output[b, oc, oh, ow]
            for ic in Segments(0, IC, CORE_IC):
              wSeg = Infer.get_w_segment(oc)
              wSeg[1] = ic  # add slice in ic
              weightTile = weight[wSeg, reuse]
              imageSeg = Infer.get_input_segment(b, oc, oh, ow)
              imageSeg[1] = ic  # add slice in ic
              imageTile = image[imageSeg, reuse]
              outputTile += TensorCore(imageTile, weightTile)
              if reuse is False:
                reuse = True  # reuse same buffer.

  assert (np.allclose(output._array, targetOutput, atol=1e-5))
  print(prefix, "total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()

demo3-3 total loaded : 4825088

```


此时对于weights的加载比之前减少了一倍, 但是如果此时OH/OW有切分, 那么同一份ic的weights也会被多次加载. 而需要注意的是SRAM的大小有限, 所以必须缩小OC上的tile size, 来保证当前策略较优. 这就是tile size与SRAM大小共同影响compute schedule.

# 4. 尝试stream input

我们还可以尝试移动image stage到外层的循环, 因为每个oc内都加载了全部的input image, 那么将image移动到外层循环就可以减少许多重复的数据加载.

```python

def demo4(imageArr: np.ndarray, weightArr: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  weight = TarcedArray(weightArr)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  IC = imageArr.shape[1]
  for b in Segments(0, B, 1):
    with GlobalHierarchy(L2SIZE):
      reuse = False
      imageTile = image[Infer.get_input_segment(b, slice(0, OC), slice(0, OH), slice(0, OW))]
      for oc in Segments(0, OC, 8):
        for oh in Segments(0, OH, OH):
          for ow in Segments(0, OW, OW):
            outputTile = output[(b, oc, oh, ow), reuse]
            for ic in Segments(0, IC, CORE_IC):
              wSeg = Infer.get_w_segment(oc)
              wSeg[1] = ic  # add slice in ic
              weightTile = weight[wSeg, reuse]
              outputTile += TensorCore(imageTile[:, ic, :, :], weightTile)
              if reuse is False:
                reuse = True  # reuse same buffer.

  assert (np.allclose(output._array, targetOutput, atol=1e-5))
  print("demo4 total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()

demo4 We can't move load image statement out of OC!

```

但是很可惜这个卷积的image比较大, 如果所以移动到外循环会因为SRAM存不下而报错. 这就是compute schedule与不同的tile size可以相互影响的情况.

# 5. 尝试进行Elemwise算子的Layer Fusion.

## 5.1 单独执行每个算子

以上实验了单层卷积的情况, 我们尝试了调整tile size/调整buffer stage的位置来减少总的数据加载次数.
接下来我们需要考虑多个算子fusion的情况, 假设卷积前面不是一个带有reduction的算子, 比如binary add, 首先单独执行两个算子. 

```python

def demo5_1(imageArr: np.ndarray, biasArr: np.ndarray, weightArr: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  bias = TarcedArray(biasArr)
  mid = TarcedArray(np.zeros_like(imageArr))
  # binary add
  (B, C, H, W) = imageArr.shape
  for b in Segments(0, B, 1):
    for c in Segments(0, C, 8):
      for h in Segments(0, H, H):
        for w in Segments(0, W, W):
          with GlobalHierarchy(L2SIZE):
            imageTile = image[b, c, h, w]
            biasTile = bias[b, c, h, w]
            midTile = mid[b, c, h, w]
            midTile += imageTile + biasTile

  demo3_3(mid._array, weightArr, outputArr, targetOutput, "demo5-1")

demo5-1 total loaded : 30515200

```


那么每次算子执行结束后, 数据需要出DDR再回到SRAM, 这样就消耗了许多带宽.


## 5.2 执行Fusion后的算子

我们可以发现,后面卷积的循环[B,OH,OW,IC]分别可以对应前面binary的[B,H,W,C], 其实即前面binary的H与W是可以依据卷积的tile size来确定, 他的C维度依据卷积的IC维度确定, 并且因为这个binary计算时没有元素依赖关系, 所以简单调整他的循环顺序我们就可以进行算子Fusion了, 也就是在IC的循环中计算elemwise的计算操作.

```python

def demo5_2(imageArr: np.ndarray, biasArr: np.ndarray, weightArr: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  bias = TarcedArray(biasArr)
  weight = TarcedArray(weightArr)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  IC = imageArr.shape[1]
  for b in Segments(0, B, 1):
    for oc in Segments(0, OC, 8):
      for oh in Segments(0, OH, OH):
        for ow in Segments(0, OW, OW):
          with GlobalHierarchy(L2SIZE):
            reuse = False
            outputTile = output[b, oc, oh, ow]
            for ic in Segments(0, IC, CORE_IC):
              wSeg = Infer.get_w_segment(oc)
              wSeg[1] = ic  # add slice in ic
              weightTile = weight[wSeg, reuse]
              imageSeg = Infer.get_input_segment(b, oc, oh, ow)
              imageSeg[1] = ic  # add slice in ic
              imageTile = image[imageSeg, reuse]
              biasTile = bias[imageSeg, reuse]
              outputTile += TensorCore(imageTile + biasTile, weightTile)
              if reuse is False:
                reuse = True  # reuse same buffer.

  assert (np.allclose(output._array, targetOutput, atol=1e-5))
  print("demo5-2 total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()

demo5-2 total loaded : 8036352


```


可以发现减少了两倍的数据搬运.

# 6. 尝试进行非Elemwise算子的Layer Fusion.

## 6.1 单独执行每个算子

首先测试两层卷积单独执行的数据加载.

```python

def demo6(Infer1: Conv2dBoundsInfer, Infer2: Conv2dBoundsInfer, imageArr: np.ndarray, weightArr1: np.ndarray, weightArr2: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  weight1 = TarcedArray(weightArr1)
  tempOutput = TarcedArray(np.zeros(Infer2.in_shape).astype(np.float32))
  (B, OC, OH, OW) = Infer2.in_shape
  with GlobalHierarchy(L2SIZE):
    reuse = False
    weight1Tile = weight1[:, :, :, :]
    for b in Segments(0, B, 1):
      for oc in Segments(0, OC, 16):
        for oh in Segments(0, OH, 48):
          for ow in Segments(0, OW, OW):
            outputTile = tempOutput[(b, oc, oh, ow), reuse]
            imageSeg = Infer1.get_input_segment(b, oc, oh, ow)
            imageTile = image[imageSeg, reuse]
            outputTile += TensorCore(imageTile, weight1Tile[oc])
            if reuse is False:
              reuse = True  # reuse same buffer.

  weight2 = TarcedArray(weightArr2)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  with GlobalHierarchy(L2SIZE):
    reuse = False
    weight2Tile = weight2[:, :, :, :]
    for b in Segments(0, B, 1):
      for oc in Segments(0, OC, 16):
        for oh in Segments(0, OH, 48):
          for ow in Segments(0, OW, OW):
            outputTile = output[(b, oc, oh, ow), reuse]
            imageSeg = Infer2.get_input_segment(b, oc, oh, ow)
            imageTile = tempOutput[imageSeg, reuse]
            outputTile += TensorCore(imageTile, weight2Tile[oc])
            if reuse is False:
              reuse = True  # reuse same buffer.

  assert (np.allclose(output._array, targetOutput, atol=1e-5))
  print("demo6 total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()

demo6 total loaded : 722528

```

## 6.2 执行Fusion后的算子

假设我们遇到前面一个算子是带有reduction的情况, 比如卷积+卷积. 那么只需要考虑将两层卷积的循环直接合并即可, 同时现在的compute schedule就不能和单层卷积时相同了, 在两层卷积的循环直接合并时, 我们无法在最后一层卷积的input channel上切分, 因为后一个卷积的每一份input channel都依赖前面一个卷积的所有input channel, 这样切分会导致前面的卷积的weights反复加载, 目前我实现的多层卷积的合并必须要在SRAM中可以存下所有的weights才可以.

```python
def demo6_1(Infer1: Conv2dBoundsInfer, Infer2: Conv2dBoundsInfer, imageArr: np.ndarray, weightArr1: np.ndarray, weightArr2: np.ndarray, outputArr: np.ndarray, targetOutput: np.ndarray):
  image = TarcedArray(imageArr)
  weight1 = TarcedArray(weightArr1)
  weight2 = TarcedArray(weightArr2)
  output = TarcedArray(outputArr)
  (B, OC, OH, OW) = outputArr.shape
  IC = imageArr.shape[1]
  with GlobalHierarchy(L2SIZE):
    reuse = False
    weight1Tile = weight1[:, :, :, :]
    weight2Tile = weight2[:, :, :, :]
    for b in Segments(0, B, 1):
      for oc in Segments(0, OC, 16):
        for oh in Segments(0, OH, 48):
          for ow in Segments(0, OW, OW):
            outputTile = output[(b, oc, oh, ow), reuse]
            imageSeg2 = Infer2.get_input_segment(b, oc, oh, ow)
            imageSeg1 = Infer1.get_input_segment(
                imageSeg2[0], imageSeg2[1], imageSeg2[2], imageSeg2[3])
            imageTile1 = image[imageSeg1, reuse]
            imageTile2 = TensorCore(imageTile1, weight1Tile)
            outputTile += TensorCore(imageTile2, weight2Tile[oc])
            if reuse is False:
              reuse = True  # reuse same buffer.

  assert (np.allclose(output._array, targetOutput, atol=1e-5))
  print("demo6-1 total loaded :", GlobalHierarchy.TotalLoaded)
  GlobalHierarchy.Reset()

demo6-1 total loaded : 206432

```

# 总结

宏观上, 我把一个Fused Layer内部计算(循环切分与buffer stage等)优化称为`Compute Schedule`, 而我之前的一篇[文章](https://zhuanlan.zhihu.com/p/585176512)在Fused Layer外部流水(buffer size search/ping pong buffer)优化称为`Buffer Schedule`. 我的理解是硬件架构决定了目前的`Compute Schedule`可能性, 接下来由软件来实现各种`Compute Schedule Pattern`, 后续再根据这些计算模式进行`Buffer Schedule`, 此时根据`Buffer Schedule`的结果来选择最优的`Compute Schedule`, 如此迭代才能尽量发挥硬件性能. 当然如果硬件架构给出的执行方式少, 那么对应的软件也简单, 否则硬件的灵活性大, 软件优化的难度也高. 整个系统的能力也就是软硬件协调程度的体现.

```
    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                       │
    │                                                                       │
    │                    ┌──────────────────────────────┐                   │
    │                    │                              │                   │
    │                    │                              │                   │
┌───┴────┐       ┌───────┴────────┐             ┌───────▼───────┐      ┌────▼───┐
│Hardware├───────►Compute Schedule│             │Buffer Schedule├──────►Software│
└───▲────┘       └───────▲────────┘             └───────┬───────┘      └────┬───┘
    │                    │                              │                   │
    │                    │                              │                   │
    │                    └╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘                   │
    │                                                                       │
    └╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
```

微观上, 对于一个算子来说, 根据循环顺序/变量分配时机/切分方式的不同会导致显著的性能差距. 比如卷积, 为了尽量减少重复load weights可以在OC/IC进行切分. 同时如果考虑存储足够大还可以在不同循环维度去stage local buffer, 在内层循环里进行复用. 然后还有tile size search的优化, 上面的例子可能描述的不多, 但其实每种不同的计算模式还需要与之配套的tile search逻辑. 比如多层卷积fusion时需要尽量同时增大H和W来减少overlap还是先增大W保持连续的load; 单层卷积时先增大IC维度的tile还是OC维度的tile来满足内部TensorCore的利用率; 单层卷积是选择多占用一些SRAM来stream input还是增大一些OC/IC的tile来减少循环次数, 总之这套search tile size逻辑都需要和compute schedule以及硬件特性相匹配的. 接下来还有buffer schedule优化, 也就是多个tiled block之间, 开多少块buffer进行并行流水比较合适, 不同buffer数量时需要自动安排好每块buffer的生命周期, 内外循环都有ping pong时对于buffer正确访问, 硬件对于buffer的stride/shape限制, 代码展开的时候需要考虑软件流水, 最后还需要分析buffer读写依赖自动插入同步指令等. 最后假设如果search出来的tile size不合适, 比如切分的OC/IC太小硬件利用率不高, 那么可能还需要调整到别的compute schedule来重新走一遍上述流程. 

关于手写算子与自动Fusion. 如果是对于我上文描述的那样, 有很多种不同的计算方式可供选择, 那么对于手写算子来说就需要消耗许多的精力维护很多看似相同但是无法复用的逻辑, 所以需要有一种简单的DSL描述这些过程从而加速开发的迭代过程. 比如[TVM TensorIR](https://zhuanlan.zhihu.com/p/451854416)或[exo-lang](https://exo-lang.dev). 但感觉目前已有的技术还没法做到更自动化的Fusion, 因为多层Fusion的时候, 需要处理各种Index的变化, 比如DDR上的Tensor加载到SRAM之后不均匀切分, 每个循环所占据的SRAM Buffer大小并不一样, 并且对于卷积来说还有Padding的问题需要在SRAM中的Tile上处理好. DSL本身最好是可以将中间依赖关系以及index变换隐藏在其背后, 降低编写算子时需要记忆的内容, 并且对于多个手写的代码块可以做到自动的分析循环的依赖来进行Fusion.

我在实现自动Fusion的过程中也发现了不少问题:

1. 缺乏分析循环间的依赖性的技术
  
>  我首先是构建了Tensor维度与循环依赖的表达式, 发现无法去分析循环间的依赖性, 比如DW卷积的OC维度就等于IC维度, Weights的IC维度此时依赖了OC维度, 而普通卷积的OC维度并不影响weights的IC维度, 因此就需要手动额外引入在IC维度上的TileVar.

2. 缺乏分析空间局部性的技术

>  如果可以直接从当前的Compute Schedule中发现如何移动循环或stage buffer可以减少数据重复加载, 那么可以指导Compute Schedule, 目前以上的优化方案还都是靠观察得到的.

3. 如何将硬件限制更好的描述到Fusion中
  
>  因为不同的case下总是需要为了兼容硬件bug/执行效率做出奇奇怪怪的修改. 比如我设计的规则是所有的L2上的Buffer按使用大小来申请, 但是由于硬件对于数据加载的速度问题, 有时候还需要申请更大的空间. 或者是硬件存在某种bug, 特定算子Fuse在一起时不能开启某些功能. 但是这些约束很难用一种通用的接口描述到自动生成的规则中. 只能在自动化的过程中hard code.

4. 如何更好将自动Fusion与手写算子结合
  
>  比如上面`demo4`的情况, 在SRAM有空余的时候, 我想在合适的地方stage image buffer, 对于手写算子来说可能就是移动几行代码的事情, 坐标变换就按当前的情况手写一个公式即可. 而自动优化为了通用还需要写许多的判断/转换. 

最终就是各种硬件规则限制/性能优化trick的问题把原本规整的自动Fusion代码切分的支离破碎, 因为整套逻辑都通用, 每次都需要这一处改动还需要考虑是不是会对其他应用的地方产生额外的影响, 导致我花费了更多的精力, 算是让我体会到了`worse is better`. 不过也有可能是我实现的自动Fusion的功能太弱, 后续再学习一些多面体的知识看一下能否有帮助.
