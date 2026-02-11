---
title: 关于mindspore
mathjax: true
toc: true
categories:
  - 深度学习
date: 2020-09-19 21:50:13
tags:
- mindspore
---

记录一些关于mindspore代码的东西。


<!--more-->

## 算子调用流程


例如GPU算子`Argmax`，他的调用顺序是：
`Argmax`<-`Primitive.Argmax`<-`ccsrc/backend/kernel_compiler/gpu/arrays/argmax_gpu_kernel.cc`<-`ccsrc/backend/kernel_compiler/gpu/cuda_impl/argmax_impl.cu`

对于`batchnorm`这类操作，因为`cudnn`中有预先写好的，因此可以不用手动写GPU impl。


## batchnorm


`batchnorm`属实比较复杂，我看了一下`ccsrc`里面有4种`batchnorm`：`[batchnorm,fused_batch_norm_ex,fused_batch_norm,batchnorm_fold,batchnorm_fold_2]`。其中`batchnorm_fold,batchnorm_fold_2`是量化时使用的。`batchnorm,fused_batch_norm_ex,fused_batch_norm`是根据当前的运行模式决定的，如果是`graph_mode`执行`P.FusedBatchNormEx`进行训练，而测试时均用`P.BatchNorm`。

目前的问题是`P.FusedBatchNormEx`和`P.BatchNorm`的`GPU`后端都不支持`2D`输入。。但我看代码其实也不复杂，因为对于`GPU`后端其实都是调用的`cudnn`库，因此需要看看为何`cudnn`不支持`2D`输入，先看看`pytorch`的底层是怎么搞的。

#### `pytorch`的`batchnorm`实现

1.  因为`pytorch`时间久，因此`aten/src/ATen/native/cudnn/BatchNorm.cpp`中包含了很多版本适配。看起来比较复杂。
2.  对于`2D`的输入他默认的`mode=CUDNN_BATCHNORM_PER_ACTIVATION`，介绍里写的是`bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice) `，然后：
   
```cpp
TensorDescriptor idesc{ *input, 4 };  // input descriptor
TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.
```
我认为他的想法应该是将`2D`输入扩展为`4D`然后调用`cudnn`的`batchnorm`。然后做了下测试，发现他们的梯度的确是相同的：
```python
import torch
import torch.nn.modules as m
import numpy as np

n1 = m.BatchNorm1d(100).to('cuda')
n1.train(True)
x = torch.randn(32, 100, device='cuda', requires_grad=True)
y = torch.randn(32, 100, device='cuda', requires_grad=True)
criterion = m.MSELoss()
y_pred = n1(x)
loss = criterion(y_pred, y)
loss.backward()
xg_1 = x.grad.cpu().numpy().copy()
x.grad.zero_()
print(xg_1)

n2 = m.BatchNorm2d(100).to('cuda')
n2.train(True)
nx = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
ny = torch.unsqueeze(torch.unsqueeze(y, -1), -1)
y_pred = n2(nx)
loss = criterion(y_pred, ny)
loss.backward()
xg_2 = x.grad.cpu().numpy().copy()
x.grad.zero_()
print(xg_2)
print(np.allclose(xg_1, xg_2))
```


#### 编译代码

编译代码首先需要按[官方配置](https://gitee.com/mindspore/mindspore/blob/master/docker/mindspore-gpu/devel/Dockerfile)设置docker，属实麻烦，因为需要下载很多依赖，必须得有`vpn`不然不行。

1.  配置`docker`使用主机的代理：我用的是`vscode`，因此在`devcontainer`中添加：

```json
	"runArgs": [
		"--runtime=nvidia",
		"--network=host"
	],
```

然后`dockerfile`添加代理地址，或者手动在终端设置：
```
RUN echo -e "\nexport https_proxy=http://127.0.0.1:8888\nexport http_proxy=http://127.0.0.1:8889" >> ~/.bashrc && source ~/.bashrc
```


2.  开始编译，icu4j校验不匹配

我切换到`0.7.0`版本，然后`./build.sh -e gpu -j 8`，一开始构建很正常，到`icu4j`就奇怪了，这个包我下载了不下5次，然后看了一下`mindspore`的源码记录：`https://gitee.com/mindspore/mindspore/commit/094bdbe253a328baee394922aeb54389ca07d563`，发现他的`md5`写错了。。。按他的更新即可。


3.	出现`No rule to make target '/usr/local/cuda/lib64/libcudnn.so', needed by 'mindspore/ccsrc/CMakeFiles/_c_expression.dir/cmake_device_link.o'.  Stop.`

这个原因是官方给的`devel`与`release`版的镜像配置不太一样，导致`libcudnn.so`没有在正确位置，执行：
```sh
ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.5 /usr/local/cuda/lib64/libcudnn.so
ln -s /usr/include/x86_64-linux-gnu/cudnn_v7.h /usr/local/cuda/include/cudnn.h
```