---
title: 使k210支持Tensorflow卷积
date: 2019-01-30 09:30:54
categories:
  - 边缘计算
tags:
-   K210
-   Tensorflow
---

我昨天咋编译模型的时候,碰到`k210 model-compiler`提示`ValueError: conv2d MobilenetV1/Conv2d_12_depthwise/depthwise:0 should use padding=SAME`.他说我的卷积输入不正确,让我使用`same`padding.但是我查看了代码,的确使用的`same`卷积.所以今天就来解决下这个问题.

<!--more-->

# 1.    查看代码

首先我看了下出错的代码:

```Python
if self.input_shape[1:3] != self.output_shape[1:3]:
    print(self.input_shape, self.output_shape)
    raise ValueError('conv2d {} should use padding=SAME'.format(tensor_info.get('name', 'noname')))
```
上面的意思是要卷积输入输出中间两维形状要相等.然后我`print`了一下我的`shape`:
```sh
[layer 23]: MobilenetV1/Conv2d_12_depthwise/Relu6:0
           shape(HWC): 15x20x256 ==> 16x20x256
           scale,bias: (0.023529411764705882,0.0) ==> (0.023529411764705882,0.0)
[5, 15, 20, 256] [5, 16, 20, 256]
```
可以看到,他这里提示15与16不匹配了.

# 2.    查看原图

因为我是强行把`mobilenet`的224输入改成了`(240,320)`的输入.所以会有一些维度上的冲突.

我查看原图的节点时发现:
```sh
[?,15,20,256]=== stride=2 padding='same' ===>[?,8,10,256]
```
# 3.    原因

这里我就想通了,在`Tensorflow`中,对于`same`padding输出为:
$$ new\_height=new\_weight=\lceil\frac{W}{S}\rceil $$
对于`vaild`输出为:
$$ new\_height=new\_weight=\lceil\frac{(W-F+1)}{S}\rceil $$

所以$\lceil15/2\rceil=8$
但是在`k210`中应该是不支持这个操作,应该是内部操作只支持整数的操作.所以我们需要修改代码.

# 4.    解决方案

1.  首先尝试padding之后用`same`卷积.
    因为这里的使用的是`depthwise_conv2d_native`卷积.所以不知道`k210`中是否支持这个卷积的`same`操作.所以先试试.
    我使用类似于下面的操作进行padding:
    ```python
    a = tf.constant(np.zeros((1, 15, 20, 256)), dtype=tf.float32)
    b = tf.space_to_batch(a, [[1, 0], [0, 0]], block_size=1)
    c = tf.nn.depthwise_conv2d_native(b, tf.ones((3, 3, 256, 1)), strides=[1, 2, 2, 1], padding='SAME')
    print(a)
    print(b)
    print(c)
    """
    Tensor("Const_3:0", shape=(1, 15, 20, 256), dtype=float32)
    Tensor("SpaceToBatchND_7:0", shape=(1, 16, 20, 256), dtype=float32)
    Tensor("DepthwiseConv2dNative_6:0", shape=(1, 8, 10, 256), dtype=float32)
    """ 
    ```
    经过尝试之后,我发现在`k210 model-compiler`报错:
    ```sh
    [layer 23]: MobilenetV1/Conv2d_12_depthwise/Relu6:0
           shape(HWC): 14x18x256 ==> 8x10x256
           scale,bias: (0.023529411764705882,0.0) ==> (0.023529411764705882,0.0)
    [5, 14, 18, 256] [5, 8, 10, 256]
    ```
    他居然把我的输入识别成了[14,18].问了群里的人,他们说`space to batch nd就是为了做padding的 后面的卷积当然不需要再次padding，所以要用valid`
    我还是不太理解为什么维度会和`Tensorflow board`里面不相同.

2.  现在尝试`valid`卷积
    那么我给`height`和`width`padding`[3,2]`.
$$ \lceil\frac{15+3-3+1}{2}\rceil=8 $$
$$ \lceil\frac{20+2-3+1}{2}\rceil=10 $$
    对应代码为:
    ```python
    tf.space_to_batch(a, [[2, 1], [1, 1]], block_size=1)
    ```
    现在来尝试一下编译.就显示编译成功了.