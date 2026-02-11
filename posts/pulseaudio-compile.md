---
title: pulseaudio交叉编译
categories:
  - 边缘计算
date: 2019-03-01 19:54:17
tags:
-   蓝牙
-   树莓派
---

交叉编译`pulseaudio`到树莓派中并调用相应`api`.但是要交叉编译一个`pulseaudio`需要先交叉编译十几个依赖,这个是非常麻烦的事情.所以我找了一个简单的方法.


<!--more-->
# 编译

1.  下载`buildroot`

我下载的是[最新的稳定版](https://buildroot.org/downloads/buildroot-2018.11.3.tar.gz)

2.  配置

为树莓派配置:
```sh
cd buildroot-2018.11.3
make raspberrypi3_defconfig
```
然后配置外部交叉编译链,参考`https://blog.csdn.net/flfihpv259/article/details/51970370`
`Toolchain path`不需要是`bin`目录的上一层.
记得`Toolchain prefix`改成`${ARCH}-linux-gnueabihf`

然后`sudo make`

3.  编译

当`buildroot`编译完成之后,在`output`目录下面的`host`文件中即包含一整套的交叉编译链,这个真的太好了~

我将`/home/zqh/Documents/buildroot-2018.11.3/output/host/bin`添加到`path`,然后使用如下命令去编译`pulseaudio`的例子即可:

```sh
arm-linux-gnueabihf-gcc testrecord.c -lpulse -lpulse-simple
```

大功告成~


