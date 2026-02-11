---
title: 重新编译Tensorflow
categories:
  - 深度学习
date: 2019-03-14 20:15:20
tags:
-   Tensorflow
---

最近搬砖赚了一个`GTX2060`,所以这两天就在折腾安装显卡以及安装`Tensorflow`.下面是我的一个记录.

<!--more-->


# 卸载旧驱动与安装新驱动

首先我要把`NVIDIA-390`驱动卸载掉,然后安装`NVIDIA-418`驱动,具体安装流程参考下面这个链接:

**注意:**前面都的参考,但是我下载的驱动版本是`418`,`cuda`版本为`10.0`,`cudnn`为`7.5.0`.

`https://blog.csdn.net/qq_33200967/article/details/80689543`

# 编译Tensorflow

首先参考[官方教程](https://www.tensorflow.org/install/source),下载什么的我就不再写了,写几个注意点.

1.  安装`bazel`版本不能超过`0.21.0`,不然还得重新装

2.  配置选项尽量参照官方的例子.

3.  如果内存小于等于8g,那么在编译的时候要加选项限制使用资源

4.  `Ubuntu`自带的`gcc-7.3`,所以编译选项要加`--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"`

5.  编译的时候发生`undefined reference to cudnnCreateLRNDescriptor@libcudnn.so.7`

    参考:
    ```sh
    sudo touch /etc/ld.so.conf.d/cuda.conf
    sudo nano /etc/ld.so.conf.d/cuda.conf
    ```
    After edit the file :
    ```sh
    /usr/local/cuda/lib64
    ```

    Save the changes :
    ```sh
    sudo ldconfig
    ```

# 最后

**NOTE:** 我安装了新显卡之后,使用`Tensorflow`必须要添加:
```sh
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
```


然后我的软件版本为`tensorflow 1.13.1,cuda 10.0,cudnn 7.5.0`,我编译的安装包在这里:(https://github.com/zhen8838/tf-linux-whell)

```sh
                          ./+o+-       zqh@pc
                  yyyyy- -yyyyyy+      OS: Ubuntu 18.04 bionic
               ://+//////-yyyyyyo      Kernel: x86_64 Linux 4.15.0-46-generic
           .++ .:/++++++/-.+sss/`      Uptime: 7h 52m
         .:++o:  /++++++++/:--:/-      Packages: 2663
        o:+o+:++.`..```.-/oo+++++/     Shell: zsh 5.4.2
       .:+o:+o/.          `+sssoo+/    Resolution: 3000x1920
  .++/+:+oo+o:`             /sssooo.   DE: GNOME 
 /+++//+:`oo+o               /::--:.   WM: GNOME Shell
 \+/+o+++`o++o               ++////.   WM Theme: 
  .++.o+++oo+:`             /dddhhh.   GTK Theme: Adwaita [GTK2/3]
       .+.o+oo:.          `oddhhhh+    Icon Theme: Adwaita
        \+.++o+o``-````.:ohdhhhhh+     Font: Cantarell 11
         `:o+++ `ohhhhhhhhyo++os:      CPU: Intel Core i7-7700 @ 8x 4.2GHz [27.8°C]
           .o:`.syhhhhhhh/.oo++o`      GPU: GeForce RTX 2060
               /osyyyyyyo++ooo+++/     RAM: 4622MiB / 16003MiB
                   ````` +oo+++o\:    
                          `oo++.   
```