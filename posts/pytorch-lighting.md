---
title: pytorch-lighting隐藏的坑
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-10-16 14:31:11
tags:
- Pytorch
- 踩坑经验
---

最近发现`pytorch-lighting`比较好用，比我在`tensorflow`里面自己写的那个好，不过因为他的结构嵌套的比较深，用起来还是会踩坑。这里来记录一下。

<!--more-->

## 使用pretrain model提取特征。

官方实例如下
```python
import torchvision.models as models

class ImagenetTransferLearning(LightningModule):
    def __init__(self):
        # init a pretrained resnet
        num_target_classes = 10
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        self.classifier = nn.Linear(2048, num_target_classes)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        ...
```

但是很不幸，在`train_step`中`self.feature_extractor`被调用的时候他的状态还是`train`的，并且由于每次模型将会被调用`self.train`被重置状态，因此如果需要一个完全固定的预训练模型需要这样，用下面这个类作为基类比较好：

```python
class PretrainNet(pl.LightningModule):
  def train(self, mode: bool):
    return super().train(False)

  def state_dict(self, destination, prefix, keep_vars):
    destination = OrderedDict()
    destination._metadata = OrderedDict()
    return destination

  def setup(self, device: torch.device):
    self.freeze()

```

我属实被他这个坑了，训练的`GAN`一直不起作用，因为预训练模型的`bn`层参数会逐渐被改变，导致越训练模型输出越趋于同一个值。