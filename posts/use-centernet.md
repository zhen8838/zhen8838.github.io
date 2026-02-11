---
title: 配置CenterNet环境
mathjax: true
toc: true
categories:
  - 深度学习
date: 2019-08-29 23:08:46
tags:
-   Pytorch
-   目标检测
---

昨天我尝试用双`cuda`的方式来配置换，但是还是遇到了`cuda`的错误，我不懂`pytorch`又没办法解决。然后我浏览下`issue`，看到有同样的问题，大概率是由于显卡是20系列的，老版本的`cuda`不行，解决方式就是升级`pytorch`版本用新的`cuda`。所以我这里把配置环境重新做个记录，免得下次又来。。


<!--more-->

# 1. 安装cuda和cudnn

这个不多说了，`cuda 10.0`和`cudnn 7.5`。

# 2. 安装python环境


1.  初始化python环境


```sh
conda create --name CenterNet python=3.6
conda activate CenterNet
conda install pytorch=1.0 torchvision cudatoolkit=10.0
```

2.  安装他所使用的库

```sh
pip install -r requirements.txt
```

3.  安装cocoapi
```sh
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
python setup.py install --user
```

4.  安装DCNv2网络

记得编译器要设置为`gcc-6`
```sh
cd src/lib/models/networks
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2.git
cd DCNv2
./make.sh
```

5. 安装nms

```sh
cd src/lib/external
make
```

# 3. 下载数据集


```sh
cd src/tools
./get_pascal_voc.sh
mv voc ../../data
```

# 4. 开始训练

```sh
python main.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --num_epochs 70 --lr_step 45,60
```

# 完成

我发现在训练模型时`gpu`使用率居然是`100%`，这个真的有点强，难道`pytorch`真的比`tensorflow`给力吗？
