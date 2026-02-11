---
title: Ubuntu多cuda版本控制
mathjax: true
toc: true
categories:
  - 工具使用
date: 2019-08-28 21:52:58
tags:
-   Tensorflow
---

为了学习`CenterNet`，配置环境弄了半天。。由于我是主用`tensorflow`的，`pytorch`搞不来，只能按他的步骤来。他的环境比较老，是`cuda 9.0 cudnn 7.1`的，然而我早就在用`cuda 10.1 cudnn 7.5`了，所以我还得安装这个版本的`cuda`。

下面我就说下安装多个版本的`cuda`的注意点。


<!--more-->


### 安装cuda 9.0

下载好了之后执行(因为我是18.04 所以要加`override`避免gcc版本不匹配的无法安装问题)：
```sh
sudo sh cuda_9.0.176_384.81_linux.run --override
```

记得安装过程中下面两点要选`no`：

1.  Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26? 

2.  Do you want to install a symbolic link at /usr/local/cuda?

### 安装cudnn

下载好了之后：
```sh
tar -xvf cudnn-9.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*
```

### 修改环境变量


改为如下：
```sh
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### 设置版本切换器：

```sh
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-9.0 40
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-10.0 50 
```

然后输入`sudo update-alternatives --config cuda`即可选择版本：

```sh
There are 2 choices for the alternative cuda (providing /usr/local/cuda).

  Selection    Path                  Priority   Status
------------------------------------------------------------
* 0            /usr/local/cuda-10.0   50        auto mode
  1            /usr/local/cuda-10.0   50        manual mode
  2            /usr/local/cuda-9.0    40        manual mode

Press <enter> to keep the current choice[*], or type selection number: 
```