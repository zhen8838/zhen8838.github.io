---
title: pytorch从任意层截断并提取数据
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-11-14 21:22:27
tags:
- Pytorch
---

我想尝试利用预训练模型的各个层的特征进行重构并检查效果，但是对于任意的已经训练好的模型，我无法修改其`forward`流程，这个时候我们想到了利用`hook`函数。使用`hook`之后，我们可能需要提取中间层的输出，但模型还是运行所有，造成了不必要的时间浪费，因此需要想一个办法在`hook`的同时对模型进行截断。

<!--more-->

# 解决方案

幸好`python`与`pytorch`均具备强大的动态特性，我们可以利用异常处理达到想要的效果，如下`demo`代码所示：
1.  首先将原始预训练模型用新模型包裹，将`forward`流程封装为`_forward_impl`
2.  接下来获取子类对象的句柄，大家可以用`model.named_children()`，这里我自己魔改了一下，跳过了一些层。
3.  为对应层添加`hook`，并且抛出异常。
4.  覆盖模型`forward`函数，处理异常。

可惜，魔改代码一时爽，适配起来就想哭。。想要一次性写出灵活性强的代码是真的难，现在还得回去把之前的所有预训练模型特征提取的代码都修改一下..
```python
def dev_get_pretrained_model_name():
  from networks.pretrainnet import Res18FaceLandmarkPreTrained, named_basic_children
  import types
  md = Res18FaceLandmarkPreTrained('models/facelandmark_full.pth')
  md.setup('cpu')
  named_basic = named_basic_children(md)

  x = torch.rand(4, 3, 256, 256)
  y = md(x)
  print(y.shape)  # torch.Size([4, 5, 2])

  # add hook
  features = []

  def hook(module: nn.Module, input: torch.Tensor):
    features.append(input[0])
    raise StopIteration
  named_basic[5][1].register_forward_pre_hook(hook)
  print(named_basic[5])
  """ 
  ('BatchNorm2d-5', BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) 
  """

  def new_forward(self, x):
    try:
      y = self._forward_impl(x)
    except StopIteration as e:
      return features[0]
    return y
  md.forward = types.MethodType(new_forward, md)

  y = md(x)
  print(y.shape)  # torch.Size([4, 64, 64, 64])
```